//! Stored tables: a streamed gathered table whose slots the host fills
//! (`TableFill::Stored`, an n-gram table), converting each row it reads from
//! the artifact into the table's resident representation as it writes it
//! (`row_conversion`), while the steps that read them wait at a gate.
//!
//! The slots and the table's directory live in host-shared storage
//! (`seismic::Tensor::shared`), which the host writes in place while device
//! work is queued. Every step that reads the table carries two timeline
//! nodes (`stored_table_rows`): a gate before its read, and a signal after
//! it. Steps are numbered in queue order, and step `n` binds both at `n`:
//!
//! - the filler (a thread of the table's own) takes each step's rows in
//!   queue order, as the host knows them or once it learns them (a step
//!   whose tokens the device selected), converts every row the step names
//!   that no slot holds into a free slot, publishes the directory entries,
//!   and raises the gate to `n`;
//! - the device raises the read timeline to `n` once step `n` has read its
//!   rows.
//!
//! A slot is free when the latest step that named its row has read it
//! (the read timeline reached that step), never when a step still queued
//! may read it; the clock takes the free slot whose row was used least
//! lately. A directory change never moves another row's entry (a row that
//! leaves its bucket leaves a tombstone, `SlotDirectory`), so a step reading
//! the directory while the filler writes it finds every row it names; the
//! directory is rebuilt without tombstones only while no step reads it.
//!
//! A row that cannot be read fails the batch slots (sequences) that named
//! it, and a failure that is no row's (the step's selection never reached
//! the host, no slot came free) fails every slot: the failure is kept for
//! the step's owner (`failure`), who discards the failed slots' outputs and
//! fails their sequences, and the gate opens all the same so the work
//! behind it drains. The other slots' rows are published and read as
//! usual. Rows are read without filling the file cache (`Reader`,
//! `CachePolicy::Uncached`).
//!
//! A row as stored is read from one range of each of its tensors
//! (`RowRange`, `import_transforms::row_tensors`: the table's codes, and its
//! scale grid's row where it has one), each in any file, and assembled by
//! the importer's own transforms (`import_transforms::assemble_row`: under a
//! grid planar, the row's codes then its scales) before it is converted.
//!
//! A request's prompt rows are prefetched (`StoredTable::prefetch`) when its
//! state opens: the filler reads them into free slots while no step waits,
//! one step's rows at a time, and holds each until a step that names it has
//! read it, or the request ends (`StoredTable::release`). The table counts
//! the rows its filler read for each step after the step was queued
//! (`TableWaits`).

use crate::device_table::KeptRows;
use crate::planning::DirectoryForm;
use crate::slot_directory::SlotDirectory;
use magnitude_family_contracts::WeightDescriptor;
use seismic_io::{Reader, Ended, Range, Want};
use seismic::cpu::repack::RowConversion;
use seismic::timeline::{Gate, Reached};
use seismic::Element;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::mpsc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

/// One stored tensor's range of every row of a stored table: row `r`'s
/// `bytes` from `first + r·bytes` in the file `source` reads.
pub struct RowRange {
    pub source: Reader,
    pub first: u64,
    pub bytes: usize,
}

/// What a stored table's steps read after they were queued: the rows its
/// filler read for a step, which the step's gate waited on. A step of rows
/// the host knew is waited on only for rows no prefetch held; a step of rows
/// the device selected reads its rows once its notice reaches the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableWaits {
    /// Steps of host-known rows, those of them that read rows after they were
    /// queued, and those rows.
    pub known_steps: u64,
    pub known_waited: u64,
    pub known_rows: u64,
    /// Steps of selected rows, and the rows read for them.
    pub selected_steps: u64,
    pub selected_rows: u64,
}

/// What a prefetch's waiter waits on: the filler drops its end once it has
/// taken the prefetch, or stopped it.
pub struct Prefetched(mpsc::Receiver<std::convert::Infallible>);

impl Prefetched {
    /// Wait until the filler has taken the prefetch.
    pub fn wait(self) {
        // Nothing is ever sent: the sender's drop ends the wait.
        let _ = self.0.recv();
    }
}

/// What keeps a slot's row: nothing; the latest step that named it, free
/// once the read timeline reaches that step; or a prefetch's hold for steps
/// not yet queued, which no read frees: only a step that names the row, or
/// the holders' release, ends it. Only a step is a timeline value: neither
/// the absence of one nor a hold is ever compared against the timeline,
/// whose comparison (CUDA's wraps a 32-bit word) reads only values near its
/// own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Named {
    Free,
    Step(u64),
    Held,
}

impl Named {
    /// What a step names: nothing before the first step (0), which no read
    /// awaits.
    fn by(step: u64) -> Self {
        match step {
            0 => Self::Free,
            step => Self::Step(step),
        }
    }

    /// Whether the slot is free by `read`: no step named its row, or that
    /// step's read is done.
    fn free(self, read: &seismic::Timeline<Reached>) -> Result<bool, seismic::ExecutionError> {
        match self {
            Self::Free => Ok(true),
            Self::Step(step) => read.reached(step),
            Self::Held => Ok(false),
        }
    }
}

/// How the host converts a stored table's rows from `stored`, their
/// encoding in the artifact, into `resident`, the representation its slots
/// hold: the CPU backend's registered row conversion, whose rows equal every
/// device repack's. Only `rows16` slots, one row to a tile, take rows one at
/// a time; `None` for any other pair.
pub fn row_conversion(stored: Element, resident: Element) -> Option<&'static RowConversion> {
    match resident.layout() {
        seismic::Layout::Rows16 => seismic::cpu::repack::resolve(stored.name(), resident.name()),
        _ => None,
    }
}

/// How long the filler waits for steps to read the rows of slots it needs
/// before it fails the step: far beyond any step's device time.
const SLOT_WAIT: Duration = Duration::from_secs(60);

/// The steps after which the rows steps named are kept.
const KEPT_AFTER_STEPS: u64 = 256;

/// The rows one step names per batch slot, each in any order and with
/// repeats.
pub enum StepRows {
    /// Known to the host when the step is queued.
    Known(Vec<Vec<u64>>),
    /// Learned once the device selected the step's tokens: the function
    /// waits for them and returns the rows, or why it could not.
    Selected(Box<dyn FnOnce() -> Result<Vec<Vec<u64>>, String> + Send>),
}

/// Why a step's rows were not all read, and whose they were.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepFailure {
    pub slots: FailedSlots,
    pub error: String,
}

/// The batch slots of a step whose rows were not all read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailedSlots {
    /// The failure was no row's: every slot.
    Every,
    /// The slots that named a row that could not be read.
    Only(BTreeSet<usize>),
}

impl StepFailure {
    fn every(error: String) -> Self {
        Self {
            slots: FailedSlots::Every,
            error,
        }
    }

    /// Whether batch slot `slot` failed.
    pub fn fails(&self, slot: usize) -> bool {
        match &self.slots {
            FailedSlots::Every => true,
            FailedSlots::Only(slots) => slots.contains(&slot),
        }
    }
}

/// A queued step's timeline bindings: its gate and read nodes bind the
/// table's timelines at `value`.
pub struct StoredStep {
    pub gate: seismic::Timeline<Gate>,
    pub read: seismic::Timeline<Reached>,
    pub value: u64,
}

/// Where the steps that read selected tokens copy them for the host
/// (`EntryNotice`): two host-shared `[rows, 2]` copies, which the steps take
/// in turn, and a timeline each step raises to its value once its copy is
/// written. Step `v` writes copy `v % 2`, which step `v + 2` writes next:
/// it does so only once the gate step `v` waits at has opened, after the
/// filler read the copy.
pub struct SelectionNotice {
    copies: seismic::Tensor,
    timeline: seismic::Timeline<Reached>,
    /// The last step's value.
    next: Mutex<u64>,
}

/// One step's selection notice: the copy it writes and its value.
pub struct NoticeStep {
    pub tokens: seismic::Tensor,
    pub timeline: seismic::Timeline<Reached>,
    pub value: u64,
}

impl SelectionNotice {
    /// The I32 extents of a notice's copies for steps of at most `rows` rows.
    pub fn extents(rows: u64) -> [u64; 3] {
        [2, rows, 2]
    }

    /// The notice's copies: what it holds.
    pub fn copies(&self) -> &seismic::Tensor {
        &self.copies
    }

    /// A notice for steps of at most `rows` rows on `device`.
    pub fn new(device: &seismic::Device, rows: u64) -> Result<Self, String> {
        let failed = |error: &dyn std::fmt::Display| format!("selection notice: {error}");
        Ok(Self {
            copies: seismic::Tensor::words(device, seismic::Element::i32(), &Self::extents(rows))
                .map_err(|error| failed(&error))?,
            timeline: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            next: Mutex::new(0),
        })
    }

    /// The notice of the next step that reads `rows` selected tokens.
    pub fn next(&self, rows: u64) -> Result<NoticeStep, String> {
        let mut next = self.next.lock().expect("selection notice lock");
        let value = *next + 1;
        let copy = value % 2;
        let tokens = self
            .copies
            .slice_leading(copy, copy + 1)
            .and_then(|copy| copy.reshape(&self.copies.extents()[1..]))
            .and_then(|copy| copy.slice_leading(0, rows))
            .map_err(|error| format!("selection notice: {error}"))?;
        *next = value;
        Ok(NoticeStep {
            tokens,
            timeline: self.timeline.clone(),
            value,
        })
    }
}

impl NoticeStep {
    /// The step's selected (token, status) rows, once the device has copied
    /// them.
    pub fn read(&self) -> Result<Vec<[i32; 2]>, String> {
        if !self.timeline.wait(self.value, SLOT_WAIT).map_err(|error| format!("the step's selected tokens: {error}"))? {
            return Err("the step's selected tokens never reached the host".into());
        }
        let mut bytes = vec![0u8; self.tokens.byte_len() as usize];
        // SAFETY: the device wrote the copy before the signal waited for;
        // the next step to write it waits for a gate this step's filler
        // opens after this read.
        unsafe { self.tokens.read_shared(0, &mut bytes) }.map_err(|error| error.to_string())?;
        Ok(bytes
            .chunks_exact(8)
            .map(|pair| {
                [
                    i32::from_le_bytes(pair[..4].try_into().expect("four bytes")),
                    i32::from_le_bytes(pair[4..].try_into().expect("four bytes")),
                ]
            })
            .collect())
    }
}

/// A table whose slots the host fills, converting its rows as stored.
pub struct StoredTable {
    shared: Arc<Shared>,
    filler: Option<JoinHandle<()>>,
    /// The last step queued.
    queued: Mutex<u64>,
    /// What resizes reserved slots (`StoredSlots::Reserved`); none of slots
    /// allocated whole.
    resizer: Mutex<Option<seismic::Resizer>>,
    /// Whether the kept rows were written as the load settled.
    settled: crate::settle::Once,
}

struct Job {
    step: u64,
    rows: StepRows,
}

/// What the filler is given: steps, taken in queue order before anything
/// else; and prefetches and releases, in order, taken while no step waits.
#[derive(Default)]
struct Jobs {
    steps: VecDeque<Job>,
    background: VecDeque<Background>,
    /// The table dropped: the filler takes the steps queued, and stops.
    closed: bool,
}

enum Background {
    /// Hold `rows` for `holder`, at most `reserve` of them at a time, and
    /// leave `reserve` slots unheld: one step's rows.
    Prefetch {
        holder: u64,
        rows: Vec<u64>,
        reserve: usize,
        done: mpsc::Sender<std::convert::Infallible>,
    },
    /// `holder` ended: its holds go.
    Release { holder: u64 },
}

/// The filler's next work.
enum Work {
    Step(Job),
    Background(Background),
}

/// Where a stored table's slots lie: `[records, record_rows, width]` rows,
/// resident, allocated whole, or reserved with their records committed
/// from the first (`StoredTable::resize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredSlots {
    Whole { records: u64, record_rows: u64 },
    Reserved { records: u64, record_rows: u64 },
}

impl StoredSlots {
    fn shape(self) -> (u64, u64, bool) {
        match self {
            Self::Whole { records, record_rows } => (records, record_rows, false),
            Self::Reserved { records, record_rows } => (records, record_rows, true),
        }
    }
}

/// How many of a record's rows the table's LRU stack samples
/// ([`RowStack`]): each record's share of a curve is then counted from
/// about this many rows (a relative resolution of about 1/16).
const SAMPLED_PER_RECORD: u64 = 256;

/// The LRU stack of a stored table's rows (Mattson's, on the bank's one
/// reuse-distance stack, `experts::Ranks`), sampled spatially by each row's
/// hash (SHARDS): a row is sampled with probability `p / record_rows`, `p`
/// the least of `SAMPLED_PER_RECORD` and `record_rows`, so a sampled row `k`
/// deep in the stack is about `k / p` records deep. Every step's rows are
/// ranked; those of the steps whose rows the device selected (decode) are
/// counted into the curve by their depth before it. A row deeper than every
/// record is dropped, and reads again as one never read.
struct RowStack {
    records: usize,
    record_rows: u64,
    /// Sampled rows by the step that last read them, at most every
    /// record's, and the steps read.
    ranks: crate::experts::Ranks<u64, u64>,
    last: std::collections::HashMap<u64, u64>,
    steps: u64,
    curve: crate::assessment::TableCurve,
}

impl RowStack {
    fn new(records: u64, record_rows: u64, ranges: Vec<crate::assessment::TableRange>) -> Self {
        Self {
            records: records as usize,
            record_rows,
            ranks: crate::experts::Ranks::default(),
            last: std::collections::HashMap::new(),
            steps: 0,
            curve: crate::assessment::TableCurve {
                steps: 0,
                rows: 0,
                sampled: vec![0; records as usize + 1],
                record_rows,
                ranges,
            },
        }
    }

    /// Step rows `rows`, each once, read together; `counted`: a decode
    /// step's, counted into the curve.
    fn read(&mut self, rows: &[u64], counted: bool) {
        let record_rows = self.record_rows;
        // A record of fewer rows has every one sampled.
        let per = SAMPLED_PER_RECORD.min(record_rows);
        let sampled = rows
            .iter()
            .copied()
            .filter(|&row| (u64::from(crate::slot_directory::hash(row as i32)) * record_rows) >> 32 < per)
            .collect::<Vec<_>>();
        let depths = sampled
            .iter()
            .map(|&row| match self.last.get(&row) {
                Some(&last) => ((self.ranks.above(last, row) as u64 / per) as usize).min(self.records),
                None => self.records,
            })
            .collect::<Vec<_>>();
        let stamp = self.steps;
        self.steps += 1;
        for &row in &sampled {
            if let Some(last) = self.last.insert(row, stamp) {
                self.ranks.remove(last, row);
            }
            self.ranks.insert(stamp, row, self.last.len());
        }
        while self.last.len() > self.records * per as usize {
            let (oldest, deepest) = self
                .ranks
                .oldest()
                .map(|(stamp, rows)| (stamp, rows[0]))
                .expect("a held row is ranked");
            self.ranks.remove(oldest, deepest);
            self.last.remove(&deepest);
        }
        if counted {
            self.curve.steps += 1;
            self.curve.rows += rows.len() as u64;
            for depth in depths {
                self.curve.sampled[depth] += 1;
            }
        }
    }
}

struct Shared {
    /// `[records, record_rows, width]` rows, resident (`StoredSlots`).
    slots: seismic::Tensor,
    /// Where the slots' storage lays each record.
    placement: seismic::RecordPlacement,
    /// The rows of a record: slot `s` is row `s % record_rows` of record
    /// `s / record_rows`.
    record_rows: u64,
    /// The rows steps read, ranked by recency.
    stack: Mutex<RowStack>,
    /// `[entries, fields]` I32 (`DirectoryForm`).
    directory: seismic::Tensor,
    fields: u64,
    /// The table's weight, whose transforms assemble a row as stored from
    /// its ranges (`import_transforms::assemble_row`).
    descriptor: WeightDescriptor,
    /// How a row as stored becomes a slot's row.
    conversion: &'static RowConversion,
    width: u64,
    /// The bytes of a slot's row.
    slot_bytes: usize,
    /// The table's `rows` rows in the artifact's files: a range of each of
    /// its row tensors.
    ranges: Vec<RowRange>,
    rows: u64,
    gate: seismic::Timeline<Gate>,
    read: seismic::Timeline<Reached>,
    jobs: Mutex<Jobs>,
    /// Wakes the filler when jobs arrive.
    arrived: Condvar,
    state: Mutex<State>,
    failures: Mutex<BTreeMap<u64, StepFailure>>,
    waits: Mutex<TableWaits>,
    /// The last step the filler took, and the last whose gate it opened.
    taken: AtomicU64,
    opened: AtomicU64,
    kept: Option<KeptRows>,
}

struct State {
    /// The leading records whose slots hold rows: every one committed.
    records: u64,
    directory: SlotDirectory,
    /// Per slot, what keeps its row.
    named: Vec<Named>,
    /// Per slot, whether a step named its row, here or when the table was
    /// last served.
    served: Vec<bool>,
    /// The rows prefetches hold (their slots `Named::Held`), each with the
    /// holders holding it.
    holds: HashMap<u64, Vec<u64>>,
}

impl StoredTable {
    /// A table of `rows` rows of `width` values of `descriptor`'s weight,
    /// stored as `stored` once assembled from `ranges` (one of each of its
    /// row tensors, `import_transforms::row_tensors`, in order), in
    /// `layout`'s slots of the `resident` representation, every record
    /// committed, and a directory of `form`, on `device`. Its slots are
    /// empty (`seed`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &seismic::Device,
        descriptor: &WeightDescriptor,
        stored: seismic::Element,
        resident: seismic::Element,
        rows: u64,
        width: u64,
        layout: StoredSlots,
        form: DirectoryForm,
        ranges: Vec<RowRange>,
        kept: Option<KeptRows>,
    ) -> Result<Self, String> {
        let failed = |error: &dyn std::fmt::Display| format!("stored table: {error}");
        let conversion = row_conversion(stored, resident).ok_or_else(|| {
            failed(&format!(
                "no host row conversion of {} into {}",
                stored.name(),
                resident.name()
            ))
        })?;
        let row_bytes = |element: seismic::Element| {
            element
                .canonical_byte_len(&[1, width])
                .map_err(|error| failed(&error))
        };
        let (slot_bytes, row_bytes) = (row_bytes(resident)?, row_bytes(stored)?);
        // A stored row and a slot's row are one piece of the conversion.
        let values = usize::try_from(width).map_err(|error| failed(&error))?;
        let piece = (conversion.source_bytes(1, values), conversion.stored_row_bytes(values));
        if piece != (row_bytes as usize, Some(slot_bytes as usize)) {
            return Err(failed(&"a stored row does not convert into one slot's row"));
        }
        // Each range lies in its file, and the ranges of a row assemble into
        // one row as stored.
        if ranges.iter().any(|range| range.source.size() < range.first + rows * range.bytes as u64) {
            return Err(failed(&"the artifact does not hold the table's rows"));
        }
        let zeros = ranges.iter().map(|range| vec![0u8; range.bytes]).collect::<Vec<_>>();
        let assembled = crate::import_transforms::assemble_row(
            descriptor,
            width,
            &zeros.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .map_err(|error| failed(&error))?;
        if assembled.len() as u64 != row_bytes {
            return Err(failed(&format!(
                "ranges of {:?} bytes assemble into {} bytes, not a row of {row_bytes}",
                ranges.iter().map(|range| range.bytes).collect::<Vec<_>>(),
                assembled.len()
            )));
        }
        let (records, record_rows, reserved) = layout.shape();
        let slots = records * record_rows;
        if slots == 0 || slots > rows || slots > i32::MAX as u64 || rows > i32::MAX as u64 {
            return Err(failed(&format!("{slots} slots for {rows} rows")));
        }
        let directory = SlotDirectory::new(rows, slots, 1, form);
        let (entries, fields) = (form.entries(rows, slots), form.fields());
        let extents = [records, record_rows, width];
        let (slots_tensor, resizer) = match reserved {
            true => seismic::Tensor::reserved_in(crate::planning::BankTier::Bank.contract_on(device), device, resident, &extents, records, None)
                .map(|(tensor, resizer)| (tensor, Some(resizer))),
            // `stored_rows` reads the slots through a record port: the table
            // of their records is held from now (a reservation holds its own).
            false => seismic::Tensor::shared(device, resident, &extents)
                .and_then(|tensor| tensor.hold_record_table().map(|()| (tensor, None))),
        }
        .map_err(|error| failed(&error))?;
        let directory_tensor =
            seismic::Tensor::words(device, seismic::Element::i32(), &[entries, fields])
                .map_err(|error| failed(&error))?;
        let placement = slots_tensor
            .record_placement()
            .ok_or_else(|| failed(&"the slots hold no records"))?;
        let curve_ranges = ranges
            .iter()
            .map(|range| crate::assessment::TableRange { first: range.first, bytes: range.bytes as u64 })
            .collect();
        let shared = Arc::new(Shared {
            slots: slots_tensor,
            placement,
            record_rows,
            stack: Mutex::new(RowStack::new(records, record_rows, curve_ranges)),
            directory: directory_tensor,
            fields,
            descriptor: descriptor.clone(),
            conversion,
            width,
            slot_bytes: slot_bytes as usize,
            ranges,
            rows,
            gate: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            read: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            jobs: Mutex::new(Jobs::default()),
            arrived: Condvar::new(),
            state: Mutex::new(State {
                records,
                directory,
                named: vec![Named::Free; slots as usize],
                served: vec![false; slots as usize],
                holds: HashMap::new(),
            }),
            failures: Mutex::new(BTreeMap::new()),
            waits: Mutex::new(TableWaits::default()),
            taken: AtomicU64::new(0),
            opened: AtomicU64::new(0),
            kept,
        });
        // The directory starts with every entry empty.
        {
            let mut state = shared.state.lock().expect("stored table lock");
            let bytes = state.directory.entry_bytes();
            // SAFETY: no device work uses the table yet.
            unsafe { shared.directory.write_shared(0, &bytes) }.map_err(|error| failed(&error))?;
        }
        let filler = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("stored-table-filler".into())
                .spawn(move || {
                    // A filler that panics fails every step whose gate it
                    // had not opened and opens them, so none waits on it,
                    // then ends with the panic (`Drop` joins it).
                    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shared.fill())) {
                        shared.died(panic.as_ref());
                        std::panic::resume_unwind(panic);
                    }
                })
                .map_err(|error| failed(&error))?
        };
        Ok(Self {
            shared,
            filler: Some(filler),
            queued: Mutex::new(0),
            resizer: Mutex::new(resizer),
            settled: Default::default(),
        })
    }

    pub fn slots_tensor(&self) -> &seismic::Tensor {
        &self.shared.slots
    }

    /// The directory readers find a row's slot in (`gathered_row`).
    pub fn directory(&self) -> &seismic::Tensor {
        &self.shared.directory
    }

    /// What the table allocates beside its slots: its directory.
    pub fn directory_bytes(&self) -> u64 {
        self.shared.directory.storage_bytes()
    }

    /// Fill the slots with the rows steps named when the table was last
    /// served, before its first step.
    pub fn seed(&self) -> Result<(), String> {
        let mut state = self.shared.state.lock().expect("stored table lock");
        let rows = self
            .shared
            .kept
            .as_ref()
            .map_or_else(Vec::new, KeptRows::read)
            .into_iter()
            .filter(|row| *row < self.shared.rows)
            .take((state.records * self.shared.record_rows) as usize)
            .collect::<Vec<_>>();
        match self.shared.bring(&mut state, 0, &[rows], true) {
            Ok((None, _)) => Ok(()),
            Ok((Some(failure), _)) => Err(failure.error),
            Err(error) => Err(error),
        }
    }

    /// Hold `rows` (a request's prompt rows, in prompt order, with repeats)
    /// for `holder` ahead of the steps that will name them: the filler reads
    /// the ones no slot holds into free slots while no step waits, at most
    /// `reserve` (one step's rows) at a time, and holds each until a step
    /// that names it has read it, or `holder` is released. It opens no gate,
    /// and stops where holding more would leave fewer than `reserve` slots
    /// unheld, or no slot is free: steps read the rest.
    pub fn prefetch(&self, holder: u64, rows: Vec<u64>, reserve: usize) -> Result<Prefetched, String> {
        let (done, waiter) = mpsc::channel();
        self.shared.submit(Background::Prefetch { holder, rows, reserve: reserve.max(1), done })?;
        Ok(Prefetched(waiter))
    }

    /// `holder` ended: the rows it holds that no step named are free once no
    /// other holder holds them.
    pub fn release(&self, holder: u64) -> Result<(), String> {
        self.shared.submit(Background::Release { holder })
    }

    /// What its steps read after they were queued, so far.
    pub fn waits(&self) -> TableWaits {
        *self.shared.waits.lock().expect("stored table waits lock")
    }

    /// The records whose slots hold rows now.
    pub fn records(&self) -> u64 {
        self.shared.state.lock().expect("stored table lock").records
    }

    /// What its decode steps read, by depth ([`RowStack`]).
    pub fn curve(&self) -> crate::assessment::TableCurve {
        self.shared.stack.lock().expect("stored table stack lock").curve.clone()
    }

    /// What the slots are charged now, as Seismic measures it; none of
    /// slots no longer held.
    pub fn charged_bytes(&self) -> Option<u64> {
        self.shared.slots.observe_storage().charged_bytes()
    }

    /// Hold the slots of `records` records of reserved slots
    /// (`StoredSlots::Reserved`), at least one and at most every one; a shrink
    /// with every step queued so far read (the domain brings the device past
    /// every begun step first). A shrink takes the rows of the records it leaves
    /// out of the directory, publishes it, then releases their storage; a
    /// growth backs its records first, then fills them as steps name rows.
    /// The storage's refusal (`Ok(Err)`) leaves the table serving the slots
    /// it uses.
    pub fn resize(&self, records: u64) -> Result<Result<(), seismic::TensorError>, String> {
        let mut resizer = self.resizer.lock().expect("stored table resizer lock");
        // Held throughout: no step is queued between the check that every
        // queued step read and the shrink that relies on it.
        let queued = self.queued.lock().expect("stored table queue lock");
        let mut state = self.shared.state.lock().expect("stored table lock");
        let current = resizer.take().ok_or("a stored table allocated whole has no resize")?;
        let (next, resized) = self.resize_with(current, &mut state, *queued, records);
        *resizer = Some(next);
        resized
    }

    /// [`Self::resize`] with the resizer taken: it comes back on every path.
    fn resize_with(
        &self,
        resizer: seismic::Resizer,
        state: &mut State,
        queued: u64,
        records: u64,
    ) -> (seismic::Resizer, Result<Result<(), seismic::TensorError>, String>) {
        let extent = self.shared.slots.extents()[0];
        let unread = match records < state.records {
            true => self.shared.read.reached(queued).map(|read| !read),
            false => Ok(false),
        };
        let unread = match unread {
            Ok(unread) => unread,
            Err(error) => return (resizer, Err(format!("a stored table's queued reads: {error}"))),
        };
        if records == 0 || records > extent || unread {
            return (
                resizer,
                Err(format!(
                    "a stored table of {extent} records resizes to {records}, a shrink with every queued step read"
                )),
            );
        }
        let record_rows = self.shared.record_rows as usize;
        if records < state.records {
            let held = records as usize * record_rows;
            state.directory.limit(held);
            state.named[held..].fill(Named::Free);
            state.served[held..].fill(false);
            // The rows the released records held are no longer held.
            let State { holds, directory, .. } = &mut *state;
            holds.retain(|row, _| directory.slot(*row).is_some());
            state.records = records;
            if let Err(error) = self.shared.publish(state) {
                return (resizer, Err(error));
            }
        }
        match resizer.recommit(records) {
            Ok(next) => {
                if records > state.records {
                    state.directory.limit(records as usize * record_rows);
                    state.records = records;
                }
                (next, Ok(Ok(())))
            }
            Err(refused) => (refused.resizer, Ok(Err(refused.error))),
        }
    }

    /// A step that reads the table is being queued, with the rows it
    /// names: its timeline bindings. The step's gate opens once its rows
    /// are in their slots, after every step queued before it.
    pub fn queue(&self, rows: StepRows) -> Result<StoredStep, String> {
        let mut queued = self.queued.lock().expect("stored table queue lock");
        let step = *queued + 1;
        {
            let mut jobs = self.shared.jobs.lock().expect("stored table jobs lock");
            if jobs.closed {
                return Err("the stored table's filler has stopped".into());
            }
            jobs.steps.push_back(Job { step, rows });
        }
        self.shared.arrived.notify_one();
        *queued = step;
        Ok(StoredStep {
            gate: self.shared.gate.clone(),
            read: self.shared.read.clone(),
            value: step,
        })
    }

    /// Step `value` was queued but will never run: its read is done once
    /// every step before it has read. Precondition: no step queued after it
    /// has been submitted.
    pub fn abandon(&self, value: u64) -> Result<(), String> {
        // The filler opens the gate of every step it takes; the device
        // raises the read timeline as every step up to a value has read, and
        // with none of them pending the host raises it in the device's place
        // for the step that never runs.
        let lost = |error: seismic::ExecutionError| format!("an abandoned stored-table step's predecessors: {error}");
        let taken = self.shared.gate.wait_signalled(value, SLOT_WAIT).map_err(lost)?
            && self.shared.read.wait(value.saturating_sub(1), SLOT_WAIT).map_err(lost)?;
        if !taken {
            return Err("an abandoned stored-table step's predecessors did not read".into());
        }
        self.shared
            .read
            .reach(value)
            .map_err(|error| error.to_string())
    }

    /// Why step `value`'s rows were not all read, once its gate opened, and
    /// whose they were: the failed slots' reads found other rows, and their
    /// outputs are not to be used.
    pub fn failure(&self, value: u64) -> Option<StepFailure> {
        self.shared
            .failures
            .lock()
            .expect("stored table failures lock")
            .get(&value)
            .cloned()
    }

    /// Step `value`'s outcome was taken: its failure, and those of the
    /// steps before it, are forgotten.
    pub fn retire(&self, value: u64) {
        let mut failures = self
            .shared
            .failures
            .lock()
            .expect("stored table failures lock");
        *failures = failures.split_off(&(value + 1));
    }
}

impl Drop for StoredTable {
    /// The filler takes every step queued, so every gate a submitted step
    /// waits at opens, and stops.
    fn drop(&mut self) {
        self.shared.jobs.lock().expect("stored table jobs lock").closed = true;
        self.shared.arrived.notify_one();
        if let Some(filler) = self.filler.take() {
            let _ = filler.join();
        }
        crate::settle::Owed::settle(self);
    }
}

/// The rows its steps named are kept for the next load, once: as the load
/// settles, or else as the table drops.
impl crate::settle::Owed for StoredTable {
    fn settle(&self) {
        if self.settled.first() {
            let state = self.shared.state.lock().expect("stored table lock");
            self.shared.keep(&state);
        }
    }
}

impl Shared {
    /// Give the filler `background` work, behind every step.
    fn submit(&self, background: Background) -> Result<(), String> {
        {
            let mut jobs = self.jobs.lock().expect("stored table jobs lock");
            if jobs.closed {
                return Err("the stored table's filler has stopped".into());
            }
            jobs.background.push_back(background);
        }
        self.arrived.notify_one();
        Ok(())
    }

    /// The filler: each step in queue order, and its other work while no
    /// step waits, until the table drops with every step taken.
    fn fill(&self) {
        loop {
            let work = {
                let mut jobs = self.jobs.lock().expect("stored table jobs lock");
                loop {
                    if let Some(job) = jobs.steps.pop_front() {
                        break Work::Step(job);
                    }
                    if jobs.closed {
                        return;
                    }
                    if let Some(background) = jobs.background.pop_front() {
                        break Work::Background(background);
                    }
                    jobs = self.arrived.wait(jobs).expect("stored table jobs lock");
                }
            };
            match work {
                Work::Step(job) => self.fill_step(job),
                Work::Background(Background::Prefetch { holder, mut rows, reserve, done }) => {
                    // One step's rows at a time: a step queued meanwhile is
                    // taken before the rest.
                    let rest = rows.split_off(reserve.min(rows.len()));
                    if self.prefetch(holder, &rows, reserve) && !rest.is_empty() {
                        self.jobs
                            .lock()
                            .expect("stored table jobs lock")
                            .background
                            .push_front(Background::Prefetch { holder, rows: rest, reserve, done });
                    }
                }
                Work::Background(Background::Release { holder }) => self.release(holder),
            }
        }
    }

    /// Step `step`: hold its rows, then open its gate.
    fn fill_step(&self, Job { step, rows }: Job) {
        self.taken.store(step, Ordering::SeqCst);
        let selected = matches!(rows, StepRows::Selected(_));
        let failure = match rows {
            StepRows::Known(rows) => Ok(rows),
            StepRows::Selected(rows) => rows(),
        }
        .and_then(|rows| {
            let distinct = rows.iter().flatten().copied().collect::<BTreeSet<_>>();
            self.stack
                .lock()
                .expect("stored table stack lock")
                .read(&distinct.into_iter().collect::<Vec<_>>(), selected);
            let mut state = self.state.lock().expect("stored table lock");
            let (failure, read) = self.bring(&mut state, step, &rows, true)?;
            let mut waits = self.waits.lock().expect("stored table waits lock");
            match selected {
                true => {
                    waits.selected_steps += 1;
                    waits.selected_rows += read;
                }
                false => {
                    waits.known_steps += 1;
                    waits.known_waited += u64::from(read > 0);
                    waits.known_rows += read;
                }
            }
            drop(waits);
            if step == KEPT_AFTER_STEPS {
                self.keep(&state);
            }
            Ok(failure)
        })
        .unwrap_or_else(|error| Some(StepFailure::every(error)));
        if let Some(failure) = failure {
            self.failures
                .lock()
                .expect("stored table failures lock")
                .insert(step, failure);
        }
        // Open the gate whatever happened: the work behind it drains.
        if let Err(error) = self.gate.signal(step) {
            self.failures
                .lock()
                .expect("stored table failures lock")
                .insert(step, StepFailure::every(error.to_string()));
        }
        self.opened.store(step, Ordering::SeqCst);
    }

    /// The filler panicked with `panic`: no step is queued from now, and
    /// every step taken or queued whose gate it had not opened fails every
    /// slot, its gate opened. Locks the panic poisoned are taken as they are.
    fn died(&self, panic: &(dyn std::any::Any + Send)) {
        let reason = panic
            .downcast_ref::<&str>()
            .map(|reason| reason.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic".into());
        let error = format!("the stored table's filler died: {reason}");
        let queued = {
            let mut jobs = self.jobs.lock().unwrap_or_else(PoisonError::into_inner);
            jobs.closed = true;
            jobs.background.clear();
            jobs.steps.drain(..).map(|job| job.step).max()
        };
        let opened = self.opened.load(Ordering::SeqCst);
        let last = queued.unwrap_or(0).max(self.taken.load(Ordering::SeqCst));
        let mut failures = self.failures.lock().unwrap_or_else(PoisonError::into_inner);
        for step in opened + 1..=last {
            failures.insert(step, StepFailure::every(error.clone()));
        }
        drop(failures);
        // A gate the device cannot be told of fails its steps as above all
        // the same.
        let _ = self.gate.signal(last);
    }

    /// Hold `rows` for `holder` in slots free now ([`StoredTable::prefetch`]):
    /// the rows a slot holds are held where they are, and the others read
    /// into free slots, never waiting for one. Whether it may hold more.
    fn prefetch(&self, holder: u64, rows: &[u64], reserve: usize) -> bool {
        let mut guard = self.state.lock().expect("stored table lock");
        let state = &mut *guard;
        let slots = (state.records * self.record_rows) as usize;
        let mut seen = HashSet::with_capacity(rows.len());
        let mut absent = Vec::new();
        let mut more = true;
        // A row outside the table fails the step that names it.
        for &row in rows.iter().filter(|row| **row < self.rows && seen.insert(**row)) {
            if let Some(slot) = state.directory.use_slot(row) {
                state.named[slot as usize] = Named::Held;
                hold(&mut state.holds, row, holder);
                continue;
            }
            if state.holds.len() + absent.len() + reserve >= slots {
                more = false;
                break;
            }
            absent.push(row);
        }
        let mut taken = Vec::with_capacity(absent.len());
        let mut claimed = HashSet::with_capacity(absent.len());
        for _ in &absent {
            let (named, read) = (&state.named, &self.read);
            let kept = |slot: usize| Ok::<_, seismic::ExecutionError>(claimed.contains(&slot) || !named[slot].free(read)?);
            match state.directory.try_free(kept) {
                Ok(Some((slot, _))) => {
                    // Free until its row is read in.
                    state.named[slot] = Named::Free;
                    claimed.insert(slot);
                    taken.push(slot);
                }
                // No slot is free now, or the device cannot say: steps read
                // the rest.
                Ok(None) | Err(_) => {
                    more = false;
                    break;
                }
            }
        }
        absent.truncate(taken.len());
        // A row that cannot be read is left to its step, which fails on it.
        // A fill, wanted at no instant: no step waits on it.
        let errors = self.read_rows(&absent, &taken, Want::undated(0.));
        for ((row, slot), error) in absent.iter().zip(&taken).zip(errors) {
            if error.is_none() {
                state.directory.hold(*row, *slot, true);
                state.served[*slot] = false;
                state.named[*slot] = Named::Held;
                hold(&mut state.holds, *row, holder);
            }
        }
        // A crowded directory is rebuilt by the next step, which waits for
        // the reads before it. The directory is host-shared storage the
        // table allocated: a write to it failing is a defect.
        self.publish(state).expect("the stored table's directory is written in place");
        more
    }

    /// `holder`'s holds go: a row no other holder holds is free once every
    /// step that named it has read it.
    fn release(&self, holder: u64) {
        let mut state = self.state.lock().expect("stored table lock");
        let State { holds, directory, named, .. } = &mut *state;
        holds.retain(|row, holders| {
            holders.retain(|held| *held != holder);
            let kept = !holders.is_empty();
            if let (false, Some(slot)) = (kept, directory.slot(*row)) {
                if named[slot as usize] == Named::Held {
                    named[slot as usize] = Named::Free;
                }
            }
            kept
        });
    }

    /// Hold every row `rows` names per batch slot for step `step` (0 before
    /// the first step): rows a slot holds are named; the others are read
    /// into free slots and published. `served`: steps named the rows. The
    /// rows that could not be read are not published, and fail the batch
    /// slots that named them (`Ok(Some)`); a failure that is no row's fails
    /// the step (`Err`). Beside it, the rows it read. A step naming a row a
    /// prefetch holds takes the hold over: the row is held until it read it.
    fn bring(
        &self,
        state: &mut State,
        step: u64,
        rows: &[Vec<u64>],
        served: bool,
    ) -> Result<(Option<StepFailure>, u64), String> {
        // Each row once, with the batch slots that name it.
        let mut naming = BTreeMap::<u64, Vec<usize>>::new();
        for (slot, rows) in rows.iter().enumerate() {
            for row in rows {
                naming.entry(*row).or_default().push(slot);
            }
        }
        let mut failed = BTreeSet::new();
        let mut errors = Vec::new();
        naming.retain(|row, slots| {
            let inside = *row < self.rows;
            if !inside {
                failed.extend(slots.iter().copied());
                errors.push(format!("table row {row} is outside the table's {} rows", self.rows));
            }
            inside
        });
        let held = (state.records * self.record_rows) as usize;
        if naming.len() > held {
            return Err("a step names more rows than the table has slots".into());
        }
        let mut absent = Vec::new();
        for row in naming.keys().copied() {
            match state.directory.use_slot(row) {
                Some(slot) => {
                    if state.named[slot as usize] == Named::Held {
                        state.holds.remove(&row);
                    }
                    state.named[slot as usize] = Named::by(step);
                    state.served[slot as usize] |= served;
                }
                None => absent.push(row),
            }
        }
        // A free slot for each absent row.
        let mut taken = Vec::with_capacity(absent.len());
        let mut claimed = std::collections::HashSet::with_capacity(absent.len());
        for _ in &absent {
            let slot = loop {
                let (named, read) = (&state.named, &self.read);
                // A slot is kept while a step that names its row may still
                // read it: this step, or one whose read is not done, or not
                // known to be (the device cannot say, which fails the step).
                let kept = |slot: usize| {
                    Ok(claimed.contains(&slot) || named[slot] == Named::Step(step) || !named[slot].free(read)?)
                };
                let free = state.directory.try_free(kept).map_err(|error: seismic::ExecutionError| format!("the stored table's reads: {error}"))?;
                if let Some((slot, _)) = free {
                    break slot;
                }
                // Every slot holds a row a step still reads, or a prefetch
                // holds: wait for the oldest such read.
                let oldest = state.named[..held]
                    .iter()
                    .filter_map(|named| match *named {
                        Named::Step(named) if named != step => Some(named),
                        Named::Free | Named::Step(_) | Named::Held => None,
                    })
                    .min()
                    .ok_or("a step names more rows than the table has slots")?;
                if !self.read.wait(oldest, SLOT_WAIT).map_err(|error| format!("the stored table's reads: {error}"))? {
                    return Err("no slot of the stored table came free".into());
                }
            };
            state.named[slot] = Named::by(step);
            claimed.insert(slot);
            taken.push(slot);
        }
        // Read the absent rows into their slots, then publish those that
        // were read; a row that was not leaves its slot free.
        let unread = self.read_rows(&absent, &taken, Want::now());
        for ((row, slot), error) in absent.iter().zip(&taken).zip(unread) {
            match error {
                None => {
                    state.directory.hold(*row, *slot, true);
                    state.served[*slot] = served;
                }
                Some(error) => {
                    failed.extend(naming[row].iter().copied());
                    errors.push(format!("table row {row}: {error}"));
                }
            }
        }
        if state.directory.crowded() {
            // No step reads the directory while it is rebuilt: every step
            // before this one has read it, and this one and the ones after
            // it wait at their gates.
            if step > 1 && !self.read.wait(step - 1, SLOT_WAIT).map_err(|error| format!("the stored table's reads: {error}"))? {
                return Err("the stored table's directory could not be rebuilt".into());
            }
            state.directory.rebuild();
        }
        self.publish(state)?;
        let failure = (!failed.is_empty()).then(|| StepFailure {
            slots: FailedSlots::Only(failed),
            error: format!("stored table rows: {}", errors.join("; ")),
        });
        Ok((failure, absent.len() as u64))
    }

    /// Read `absent`'s rows as stored, a range of each of the table's row
    /// tensors per row, wanted as `want`, then assemble and convert each into
    /// its slot of `taken`: per row, why it was not, if it was not.
    fn read_rows(&self, absent: &[u64], taken: &[usize], want: Want) -> Vec<Option<String>> {
        let mut unread = vec![None; absent.len()];
        if absent.is_empty() {
            return unread;
        }
        // Every range's bytes of every absent row, a buffer per range, read
        // as one request per range, all in flight together.
        let buffers = self
            .ranges
            .iter()
            .map(|range| Arc::new(seismic_io::Buffer::new(absent.len() * range.bytes)))
            .collect::<Vec<_>>();
        let mut requests = Vec::with_capacity(self.ranges.len());
        for (range, buffer) in self.ranges.iter().zip(&buffers) {
            let reads = absent
                .iter()
                .enumerate()
                .map(|(index, row)| Range {
                    offset: range.first + row * range.bytes as u64,
                    length: range.bytes,
                    into: buffer.clone(),
                    at: index * range.bytes,
                })
                .collect();
            // SAFETY: the reads are distinct rows of `buffer`, which nothing
            // reads until the request has ended.
            match unsafe { range.source.request(reads, want) } {
                Ok(request) => requests.push(request),
                Err(error) => unread.fill(Some(error.to_string())),
            }
        }
        for request in requests {
            match request.wait() {
                Ended::Read { failed } => {
                    for (index, error) in failed {
                        unread[index].get_or_insert_with(|| error.to_string());
                    }
                }
                Ended::Panicked(panic) => panic.resume(),
            }
        }
        // SAFETY: every request that read into the buffers has ended (one
        // not made read none).
        let read = buffers.iter().map(|buffer| unsafe { buffer.bytes() }).collect::<Vec<_>>();
        let mut converted = vec![0u8; self.slot_bytes];
        for (index, slot) in taken.iter().enumerate() {
            if unread[index].is_some() {
                continue;
            }
            let pieces = self
                .ranges
                .iter()
                .zip(&read)
                .map(|(range, bytes)| &bytes[index * range.bytes..][..range.bytes])
                .collect::<Vec<_>>();
            // `new` assembled a row of these ranges.
            let stored = crate::import_transforms::assemble_row(&self.descriptor, self.width, &pieces)
                .expect("a row's ranges assemble into one row as stored");
            self.convert(&stored, &mut converted);
            // SAFETY: the slot's previous row has been read by every step
            // that named it, and no queued step names the slot's new row
            // before its gate opens.
            if let Err(error) = unsafe { self.slots.write_shared(self.slot_offset(*slot as u64), &converted) } {
                unread[index] = Some(error.to_string());
            }
        }
        unread
    }

    /// Where slot `slot` lies in the slots' storage: its row of its record,
    /// where the storage lays the record (`Tensor::record_placement`).
    fn slot_offset(&self, slot: u64) -> u64 {
        self.placement.offset(slot / self.record_rows) + (slot % self.record_rows) * self.slot_bytes as u64
    }

    /// The slot row of the row `stored` as stored (`row_conversion`).
    fn convert(&self, stored: &[u8], slot: &mut [u8]) {
        // `stored` is an assembled row, the record of a matrix of one row
        // of `width` values as stored, and `slot` a buffer of `slot_bytes`,
        // one resident row: one piece of the conversion (`Shared::new`).
        self.conversion
            .piece(self.width as usize, 1, 0..1, stored, slot)
            .expect("a stored row is one piece of the conversion into a slot's row");
    }

    /// Write the directory entries that changed.
    fn publish(&self, state: &mut State) -> Result<(), String> {
        for (first, bytes) in state.directory.take_changes() {
            // SAFETY: a change never moves the entry of a row a reading
            // step names (`SlotDirectory`); a rebuild waited for every
            // reader.
            unsafe {
                self.directory
                    .write_shared(first * self.fields * 4, &bytes)
            }
            .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Keep the rows of every slot a step named, here or when the table was
    /// last served.
    fn keep(&self, state: &State) {
        let Some(kept) = &self.kept else {
            return;
        };
        let mut rows = (0..state.named.len())
            .filter(|slot| state.served[*slot])
            .filter_map(|slot| state.directory.held(slot))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        kept.write(&rows);
    }
}

/// `holder` holds `row` (once, however often it asks).
fn hold(holds: &mut HashMap<u64, Vec<u64>>, row: u64, holder: u64) {
    let holders = holds.entry(row).or_default();
    if !holders.contains(&holder) {
        holders.push(holder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_io::CachePolicy;
    use seismic::{BackendName, Element};

    const ROWS: u64 = 4096;
    const WIDTH: u64 = 160;
    const ROW_BYTES: usize = 90;
    /// A resident `iq4g32s@rows16` row of 160 values: 80 bytes of codes, 10
    /// of scales, aligned to 16.
    const SLOT_BYTES: usize = 96;

    fn stored() -> Element {
        Element::named("gguf_iq4_nl").unwrap()
    }

    fn resident() -> Element {
        Element::named("iq4g32s@rows16").unwrap()
    }

    /// Table row `row` as stored: five IQ4_NL blocks whose scale bytes name
    /// the row and whose nibbles follow it, in the conversion's domain.
    fn row(row: u64) -> Vec<u8> {
        let bytes = (0..ROW_BYTES)
            .map(|byte| (row as usize * 7 + byte * 13) as u8)
            .collect();
        resident().repack_source(stored(), &[1, WIDTH], bytes).unwrap()
    }

    /// Table row `row` as its slot holds it: the registry's host reference
    /// of its conversion.
    fn slot_row(row: u64) -> Vec<u8> {
        resident()
            .repack_host(stored(), &[1, WIDTH], &self::row(row))
            .unwrap()
    }

    fn table(name: &str, slots: u64, form: DirectoryForm) -> (std::path::PathBuf, StoredTable) {
        let path = file(name);
        let table = StoredTable::new(
            &device(),
            &descriptor(),
            stored(),
            resident(),
            ROWS,
            WIDTH,
            StoredSlots::Whole { records: 1, record_rows: slots },
            form,
            ranges(&path),
            None,
        )
        .unwrap();
        (path, table)
    }

    /// The table's weight: its rows as stored, one range each.
    fn descriptor() -> WeightDescriptor {
        WeightDescriptor::stored("table", [ROWS, WIDTH])
    }

    /// The one range of each row of the table `file` wrote.
    fn ranges(path: &std::path::Path) -> Vec<RowRange> {
        vec![RowRange { source: Reader::open(path, CachePolicy::Uncached).unwrap(), first: 100, bytes: ROW_BYTES }]
    }

    /// The requested test backend's device, else the CPU's: off the CPU a
    /// timeline's host and device sides are apart.
    fn device() -> seismic::Device {
        crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::testing::requested_backend().unwrap_or(BackendName::Cpu))
            .unwrap()
    }

    /// A file of the table's rows after 100 bytes.
    fn file(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "magnitude-stored-table-{name}-{}.bin",
            std::process::id()
        ));
        let mut bytes = vec![0xee; 100];
        bytes.extend((0..ROWS).flat_map(row));
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    /// The slot row a reader finds for `row`, probing the device's
    /// directory as `gathered_row` does.
    fn found(table: &StoredTable, row: u64) -> Option<Vec<u8>> {
        let mut directory = vec![0u8; table.directory().byte_len() as usize];
        // SAFETY: no device work uses the table.
        unsafe { table.directory().read_shared(0, &mut directory) }.unwrap();
        let entries = directory
            .chunks_exact(4)
            .map(|entry| i32::from_le_bytes(entry.try_into().unwrap()))
            .collect::<Vec<_>>();
        let slot = match table.shared.fields {
            1 => entries[row as usize],
            _ => {
                let buckets = entries.len() / 2;
                let mut probe = crate::slot_directory::bucket(row as i32, buckets);
                loop {
                    match entries[2 * probe] {
                        key if key == row as i32 => break entries[2 * probe + 1],
                        -1 => return None,
                        _ => probe = (probe + 1) % buckets,
                    }
                }
            }
        };
        let slot = usize::try_from(slot).ok()?;
        let mut bytes = vec![0u8; SLOT_BYTES];
        // SAFETY: as above.
        unsafe { table.slots_tensor().read_shared(table.shared.slot_offset(slot as u64), &mut bytes) }
            .unwrap();
        Some(bytes)
    }

    /// A reserved table shrinks to its leading records: the rows the
    /// records it releases held leave the directory, every other row stays
    /// found, and its charge falls; a shrink with a queued step unread is
    /// refused, and a step naming more rows than the slots it uses fails.
    /// Grown again, steps fill the records it backs.
    #[test]
    fn a_reserved_table_resizes_by_records() {
        let path = file("resize");
        let device = device();
        let table = StoredTable::new(
            &device,
            &descriptor(),
            stored(),
            resident(),
            ROWS,
            WIDTH,
            StoredSlots::Reserved { records: 4, record_rows: 256 },
            DirectoryForm::Hashed,
            ranges(&path),
            None,
        )
        .unwrap();
        let rows = |count: u64, from: u64| (from..from + count).map(|row| row * 3 % ROWS).collect::<Vec<_>>();
        let run = |rows: Vec<u64>| {
            let step = table.queue(StepRows::Known(vec![rows])).unwrap();
            assert!(step.gate.wait_signalled(step.value, Duration::from_secs(10)).unwrap());
            step
        };
        // Every slot holds a row.
        let every = rows(1024, 0);
        let step = run(every.clone());
        assert_eq!(table.failure(step.value), None);
        assert!(table.resize(2).is_err(), "a shrink waits for every queued step's read");
        read_done(&step);
        let before = table.charged_bytes().unwrap();
        table.resize(2).unwrap().unwrap();
        assert_eq!(table.records(), 2);
        // It frees the whole granules the two records' bytes leave
        // (none where the four lie in one granule, as on Metal's).
        let backing = granules(&device, 256);
        let record = record_bytes(256);
        assert_eq!(
            before - table.charged_bytes().unwrap(),
            backing.backing(4 * record).unwrap() - backing.backing(2 * record).unwrap()
        );
        let kept = every.iter().filter(|row| found(&table, **row).is_some()).collect::<Vec<_>>();
        assert_eq!(kept.len(), 512);
        for row in kept {
            assert_eq!(found(&table, *row), Some(slot_row(*row)));
        }
        // The slots in use hold 512 rows a step: 513 fail it.
        let step = run(rows(513, 2000));
        assert!(table.failure(step.value).is_some());
        read_done(&step);
        table.resize(4).unwrap().unwrap();
        assert_eq!((table.records(), table.charged_bytes().unwrap()), (4, before));
        let step = run(rows(1024, 1024));
        assert_eq!(table.failure(step.value), None);
        for row in rows(1024, 1024) {
            assert_eq!(found(&table, row), Some(slot_row(row)));
        }
        read_done(&step);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// The bytes of one record of `record_rows` slots.
    fn record_bytes(record_rows: u64) -> u64 {
        resident().canonical_byte_len(&[1, record_rows, WIDTH]).unwrap()
    }

    /// The reservation the test device backs a stored table's slots in, as
    /// it reserves records of `record_rows` slots: its granule is what a
    /// resize backs and frees whole.
    fn granules(device: &seismic::Device, record_rows: u64) -> seismic::Reservation {
        let contract = crate::planning::BankTier::Bank.contract_on(device);
        let record = std::num::NonZeroU64::new(record_bytes(record_rows)).unwrap();
        device
            .info()
            .reserving(contract)
            .unwrap()
            .expect("the test device reserves a stored table's slots")
            .of_records(seismic::ReservedRecords::of(record))
    }

    /// A reserved table charges whole granules of its device's reservation
    /// (read from the device): a shrink by records frees exactly the whole
    /// granules its records' bytes leave, and none for a shrink within one
    /// granule. Sized to three granules of 16-row records, at least two to a
    /// granule.
    #[test]
    fn a_reserved_table_shrink_frees_whole_granules() {
        let device = device();
        let record_rows = 16;
        let reservation = granules(&device, record_rows);
        let (granule, record) = (reservation.granule, record_bytes(record_rows));
        assert!(record * 2 <= granule, "{record}-byte records, {granule}-byte granules");
        // The most records within three granules, and within one.
        let (three, one) = (3 * granule / record, granule / record);
        let rows = three * record_rows;
        let path = std::env::temp_dir().join(format!("magnitude-stored-table-granules-{}.bin", std::process::id()));
        std::fs::File::create(&path).unwrap().set_len(100 + rows * ROW_BYTES as u64).unwrap();
        let table = StoredTable::new(
            &device,
            &WeightDescriptor::stored("table", [rows, WIDTH]),
            stored(),
            resident(),
            rows,
            WIDTH,
            StoredSlots::Reserved { records: three, record_rows },
            DirectoryForm::Hashed,
            ranges(&path),
            None,
        )
        .unwrap();
        let charged = || table.charged_bytes().unwrap();
        let full = charged();
        // One record less still lies in the third granule: nothing is freed.
        table.resize(three - 1).unwrap().unwrap();
        assert_eq!(charged(), full, "a shrink within a granule frees nothing");
        // Down to one granule's records: the other two are freed whole.
        table.resize(one).unwrap().unwrap();
        assert_eq!(full - charged(), 2 * granule, "a shrink frees whole granules");
        // Growing back backs them again.
        table.resize(three).unwrap().unwrap();
        assert_eq!(charged(), full);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// The row stack counts each decode step's rows by their depth in
    /// records: every row of a record of no more rows than it samples is
    /// sampled, and a row read `k` distinct rows ago is `k / 256` records
    /// deep; a row never read is deeper than every record. A step's cost
    /// falls as records hold more of its rows.
    #[test]
    fn the_row_stack_counts_decode_rows_by_their_depth_in_records() {
        let mut stack = RowStack::new(4, SAMPLED_PER_RECORD, vec![crate::assessment::TableRange { first: 0, bytes: 90 }]);
        for row in 0..600 {
            stack.read(&[row], false);
        }
        // Rows 0, 400 and 599 are 599, 199 and 0 deep; 700 was never read.
        stack.read(&[0, 400, 599, 700], true);
        let curve = &stack.curve;
        assert_eq!((curve.steps, curve.rows), (1, 4));
        assert_eq!(curve.sampled, [2, 0, 1, 0, 1]);
        let line = crate::assessment::ReadLine::new(1e-4, 1e-9).unwrap();
        let cost = |records| curve.step_seconds(records, &line).unwrap();
        assert!(cost(0) > cost(1) && cost(1) == cost(2) && cost(2) > cost(3) && cost(3) == cost(4));
        assert!(cost(4) > 0.);
    }

    /// Mark step `value`'s read done, as the device does after its read.
    fn read_done(step: &StoredStep) {
        step.read.reach(step.value).unwrap();
    }

    /// Rows prefetched ahead of the steps that name them are read before
    /// those steps are queued, which then read none; a step of rows no
    /// prefetch held reads its own. A prefetch leaves one step's rows of
    /// slots unheld, and holds its rows until a step that names them has
    /// read them (other steps take other slots), or its holder is released.
    #[test]
    fn prefetched_rows_are_read_before_their_steps_and_held_until_read() {
        let (path, table) = table("prefetch", 64, DirectoryForm::Hashed);
        let run = |rows: std::ops::Range<u64>| {
            let step = table.queue(StepRows::Known(vec![rows.collect()])).unwrap();
            assert!(step.gate.wait_signalled(step.value, Duration::from_secs(10)).unwrap());
            assert_eq!(table.failure(step.value), None);
            read_done(&step);
        };
        let held = |rows: &[u64]| rows.iter().filter(|row| found(&table, **row).is_some()).count();
        // Sixty rows asked, some twice: 48 held, leaving a step's 16 slots.
        let prompt = (0..60).map(|row| row * 5).collect::<Vec<_>>();
        let asked = prompt.iter().chain(&prompt[..10]).copied().collect();
        table.prefetch(1, asked, 16).unwrap().wait();
        assert_eq!(held(&prompt), 48);
        assert_eq!(held(&prompt[..48]), 48);
        assert_eq!(table.waits(), TableWaits::default());
        // Steps of other rows take the unheld slots, never a held one.
        run(1000..1016);
        run(2000..2016);
        assert_eq!(held(&prompt[..48]), 48);
        for chunk in prompt[..48].chunks(16) {
            let step = table.queue(StepRows::Known(vec![chunk.to_vec()])).unwrap();
            assert!(step.gate.wait_signalled(step.value, Duration::from_secs(10)).unwrap());
            for row in chunk {
                assert_eq!(found(&table, *row), Some(slot_row(*row)));
            }
            read_done(&step);
        }
        let waits = table.waits();
        assert_eq!((waits.known_steps, waits.known_waited, waits.known_rows), (5, 2, 32));
        // Read by their steps, they are held no more: a step takes every slot.
        run(3000..3064);
        // Nor are a released holder's rows.
        table.prefetch(2, (3500..3540).collect(), 16).unwrap().wait();
        assert_eq!(held(&(3500..3540).collect::<Vec<_>>()), 40);
        table.release(2).unwrap();
        // Taken after the release.
        table.prefetch(3, Vec::new(), 16).unwrap().wait();
        run(3600..3664);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// The E4M3 value of `code` (S.EEEE.MMM, bias 7, NaN at S.1111.111),
    /// decoded independently of the registry.
    fn e4m3(code: u8) -> f64 {
        let (exponent, mantissa) = ((code >> 3) & 15, code & 7);
        let magnitude = match (exponent, mantissa) {
            (15, 7) => f64::NAN,
            (0, _) => f64::from(mantissa) * 2f64.powi(-9),
            _ => (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7),
        };
        if code & 0x80 != 0 { -magnitude } else { magnitude }
    }

    /// A table of E4M3 rows 256 wide under a `[1, 32]` UE8M0 grid, its codes
    /// and its scales in two files (V4.1's n-gram table, `gguf_f8_e4m3_e8m0_1x32`):
    /// each slot row read through `stored_rows` decodes to exactly
    /// `code·2^(e−127)`, its codes' and its own scales', every code (NaN
    /// codes NaN) under scales from 2^−27 to 2^12, on the CPU and Metal.
    #[test]
    fn a_gridded_row_from_two_files_decodes_exactly() {
        use magnitude_family_contracts::{ImportTransform, ScaleCodes};
        use magnitude_kernels::stored_rows;
        const ROWS: u64 = 64;
        const WIDTH: u64 = 256;
        let code = |row: u64, column: u64| (row * 37 + column) as u8;
        let scale = |row: u64, group: u64| (100 + (row * 8 + group) % 40) as u8;
        let write = |name: &str, header: usize, bytes: Vec<u8>| {
            let path = std::env::temp_dir().join(format!("magnitude-gridded-{name}-{}.bin", std::process::id()));
            std::fs::write(&path, [vec![0xee; header], bytes].concat()).unwrap();
            path
        };
        let codes = write("codes", 100, (0..ROWS).flat_map(|row| (0..WIDTH).map(move |column| code(row, column))).collect());
        let scales = write("scales", 37, (0..ROWS).flat_map(|row| (0..WIDTH / 32).map(move |group| scale(row, group))).collect());
        let descriptor = WeightDescriptor {
            name: "codes".into(),
            shape: vec![ROWS, WIDTH],
            transforms: vec![ImportTransform::ScaleBlocks { tensor: "scales".into(), tile: [1, 32], codes: ScaleCodes::Ue8m0 }],
        };
        let resident = Element::named("mxfp8g32@rows16").unwrap();
        let devices = seismic::Devices::discover(&[]).unwrap();
        let mut backends = Vec::new();
        for backend in [BackendName::Cpu, BackendName::Metal] {
            let Ok(device) = crate::test_support::open_backend(&devices, backend) else {
                continue;
            };
            backends.push(backend);
            let ranges = vec![
                RowRange { source: Reader::open(&codes, CachePolicy::Uncached).unwrap(), first: 100, bytes: WIDTH as usize },
                RowRange { source: Reader::open(&scales, CachePolicy::Uncached).unwrap(), first: 37, bytes: (WIDTH / 32) as usize },
            ];
            let table = StoredTable::new(
                &device,
                &descriptor,
                Element::named("gguf_f8_e4m3_e8m0_1x32").unwrap(),
                resident,
                ROWS,
                WIDTH,
                StoredSlots::Whole { records: 1, record_rows: ROWS },
                DirectoryForm::Dense,
                ranges,
                None,
            )
            .unwrap();
            let step = table.queue(StepRows::Known(vec![(0..ROWS).collect()])).unwrap();
            // The filler opens the gate once the step's rows are written and
            // published.
            assert!(step.gate.wait_signalled(step.value, Duration::from_secs(10)).unwrap(), "{backend:?}: the filler never filled the step");
            assert_eq!(table.failure(step.value), None);
            let kernel = stored_rows::native_for_device_with(
                &device,
                stored_rows::Elements { E: resident, A: Element::f32() },
                &seismic::generated::native_implementation::<stored_rows::Entry>(&device)
                    .unwrap()
                    .unwrap()
                    .default_specialization(&seismic::NativeSpecialization::new().with_static("W", WIDTH))
                    .unwrap(),
            )
            .unwrap();
            let selections = (0..ROWS as i32).flat_map(|row| [row, 0]).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
            let tokens = seismic::Tensor::from_host(&device, Element::i32(), &[ROWS, 2], &selections).unwrap();
            let rows = kernel
                .call(stored_rows::Args { slots: table.slots_tensor(), tokens: &tokens, directory: table.directory() })
                .unwrap()
                .value
                .read_to_host()
                .unwrap();
            for (at, value) in rows.chunks_exact(4).enumerate() {
                let (row, column) = (at as u64 / WIDTH, at as u64 % WIDTH);
                let value = f32::from_le_bytes(value.try_into().unwrap());
                let expected = (e4m3(code(row, column)) * 2f64.powi(i32::from(scale(row, column / 32)) - 127)) as f32;
                assert!(
                    value.to_bits() == expected.to_bits() || value.is_nan() && expected.is_nan(),
                    "{backend:?} row {row} column {column}: {value:e}, not {expected:e}"
                );
            }
        }
        assert!(backends.contains(&BackendName::Cpu));
        assert_eq!(backends.contains(&BackendName::Metal), cfg!(target_os = "macos"), "Metal decodes where it runs");
        std::fs::remove_file(codes).unwrap();
        std::fs::remove_file(scales).unwrap();
    }

    #[test]
    fn every_step_finds_its_rows_once_its_gate_opens() {
        for form in [DirectoryForm::Dense, DirectoryForm::Hashed] {
            let (path, table) = table("rows", 64, form);
            let mut state = 5u64;
            for step in 0..200u64 {
                let rows = (0..16)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (state >> 33) % 300 * 13 % ROWS
                    })
                    .collect::<Vec<_>>();
                let queued = table
                    .queue(StepRows::Known(vec![rows[..8].to_vec(), rows[8..].to_vec()]))
                    .unwrap();
                assert_eq!(queued.value, step + 1);
                assert!(queued.gate.wait_signalled(queued.value, Duration::from_secs(10)).unwrap());
                assert_eq!(table.failure(queued.value), None);
                for row in &rows {
                    assert_eq!(found(&table, *row), Some(slot_row(*row)), "{form:?} row {row}");
                }
                read_done(&queued);
            }
            drop(table);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn a_slot_a_queued_step_reads_is_not_taken_until_that_step_has_read_it() {
        let (path, table) = table("eviction", 4, DirectoryForm::Hashed);
        let first = table.queue(StepRows::Known(vec![vec![1, 2, 3, 4]])).unwrap();
        assert!(first.gate.wait_signalled(1, Duration::from_secs(10)).unwrap());
        // The next step needs a slot while every slot holds a row the first
        // step has not read: its gate waits for that read.
        let second = table.queue(StepRows::Known(vec![vec![5]])).unwrap();
        assert!(!second.gate.wait_signalled(2, Duration::from_millis(100)).unwrap());
        for row in [1, 2, 3, 4] {
            assert_eq!(found(&table, row), Some(slot_row(row)));
        }
        read_done(&first);
        assert!(second.gate.wait_signalled(2, Duration::from_secs(10)).unwrap());
        assert_eq!(found(&table, 5), Some(slot_row(5)));
        // One of the first step's rows left; the others are still found.
        let held = [1, 2, 3, 4]
            .into_iter()
            .filter(|row| found(&table, *row).is_some())
            .count();
        assert_eq!(held, 3);
        read_done(&second);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// An abandoned step's read is done once the steps before it have read:
    /// abandoning it waits for the filler to take it (the gate the host
    /// raised) and for the device's read of the step before, then raises
    /// the read in the device's place, so the slots it named come free to
    /// the steps after it.
    #[test]
    fn an_abandoned_step_frees_its_slots_for_the_steps_after_it() {
        let (path, table) = table("abandoned", 4, DirectoryForm::Hashed);
        let first = table.queue(StepRows::Known(vec![vec![1, 2]])).unwrap();
        let abandoned = table.queue(StepRows::Known(vec![vec![3, 4]])).unwrap();
        assert!(abandoned.gate.wait_signalled(abandoned.value, Duration::from_secs(10)).unwrap());
        read_done(&first);
        table.abandon(abandoned.value).unwrap();
        assert!(abandoned.read.reached(abandoned.value).unwrap());
        // Every slot holds a row of the two steps: the next takes those of
        // both, the abandoned step's among them.
        let next = table.queue(StepRows::Known(vec![vec![5, 6, 7, 8]])).unwrap();
        assert!(next.gate.wait_signalled(next.value, Duration::from_secs(10)).unwrap());
        assert_eq!(table.failure(next.value), None);
        for row in [5, 6, 7, 8] {
            assert_eq!(found(&table, row), Some(slot_row(row)));
        }
        read_done(&next);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rows_learned_late_and_failures_still_open_the_gate() {
        let (path, table) = table("selected", 8, DirectoryForm::Dense);
        let (sender, receiver) = mpsc::channel::<Vec<Vec<u64>>>();
        let step = table
            .queue(StepRows::Selected(Box::new(move || {
                receiver.recv().map_err(|error| error.to_string())
            })))
            .unwrap();
        assert!(!step.gate.signalled(1).unwrap());
        sender.send(vec![vec![7], vec![9]]).unwrap();
        assert!(step.gate.wait_signalled(1, Duration::from_secs(10)).unwrap());
        assert_eq!(found(&table, 9), Some(slot_row(9)));
        read_done(&step);
        // A row outside the table fails the slot that named it, and the
        // step's gate opens.
        let failed = table.queue(StepRows::Known(vec![vec![2], vec![ROWS]])).unwrap();
        assert!(failed.gate.wait_signalled(2, Duration::from_secs(10)).unwrap());
        let failure = table.failure(2).unwrap();
        assert_eq!(failure.slots, FailedSlots::Only(BTreeSet::from([1])));
        read_done(&failed);
        // Rows that never reach the host fail every slot.
        let lost = table
            .queue(StepRows::Selected(Box::new(|| Err("never selected".into()))))
            .unwrap();
        assert!(lost.gate.wait_signalled(3, Duration::from_secs(10)).unwrap());
        assert_eq!(table.failure(3).unwrap().slots, FailedSlots::Every);
        read_done(&lost);
        // A failure stays until its step is retired.
        let later = table.queue(StepRows::Known(vec![vec![1]])).unwrap();
        assert!(later.gate.wait_signalled(4, Duration::from_secs(10)).unwrap());
        assert_eq!(table.failure(4), None);
        assert!(table.failure(2).is_some());
        table.retire(3);
        assert_eq!(table.failure(2), None);
        assert_eq!(table.failure(3), None);
        read_done(&later);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// A filler that panics fails every step whose gate it had not opened,
    /// the one it took and the ones queued behind it, and opens their gates:
    /// no step waits on a dead filler, and none is queued after it.
    #[test]
    fn a_filler_that_panics_fails_and_opens_every_gate_behind_it() {
        let (path, table) = table("panic", 8, DirectoryForm::Dense);
        let (go, wait) = mpsc::channel::<()>();
        let first = table
            .queue(StepRows::Selected(Box::new(move || -> Result<Vec<Vec<u64>>, String> {
                wait.recv().unwrap();
                panic!("an injected panic")
            })))
            .unwrap();
        let second = table.queue(StepRows::Known(vec![vec![1]])).unwrap();
        go.send(()).unwrap();
        for step in [&first, &second] {
            assert!(step.gate.wait_signalled(step.value, Duration::from_secs(10)).unwrap());
            let failure = table.failure(step.value).unwrap();
            assert_eq!(failure.slots, FailedSlots::Every);
            assert!(failure.error.contains("an injected panic"), "{}", failure.error);
        }
        assert!(table.queue(StepRows::Known(vec![vec![2]])).is_err());
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// A read failure (the artifact cut short under the open table) fails
    /// only the batch slot whose rows it held: the other slot's rows are
    /// published and found, and the failed row is not.
    #[test]
    fn a_read_failure_fails_only_the_slots_that_named_its_rows() {
        let (path, table) = table("injected", 64, DirectoryForm::Hashed);
        let cut = 100 + 2048 * ROW_BYTES as u64;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(cut)
            .unwrap();
        let step = table
            .queue(StepRows::Known(vec![vec![1, 2, 3], vec![3, 3500]]))
            .unwrap();
        assert!(step.gate.wait_signalled(1, Duration::from_secs(10)).unwrap());
        let failure = table.failure(1).unwrap();
        assert_eq!(failure.slots, FailedSlots::Only(BTreeSet::from([1])));
        assert!(failure.fails(1) && !failure.fails(0));
        for row in [1, 2, 3] {
            assert_eq!(found(&table, row), Some(slot_row(row)));
        }
        assert_eq!(found(&table, 3500), None);
        read_done(&step);
        // The next step reads as usual.
        let next = table.queue(StepRows::Known(vec![vec![4], vec![5]])).unwrap();
        assert!(next.gate.wait_signalled(2, Duration::from_secs(10)).unwrap());
        assert_eq!(table.failure(2), None);
        assert_eq!(found(&table, 5), Some(slot_row(5)));
        read_done(&next);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }

    /// A cold load and a warm one, seeded from the rows the cold one's steps
    /// named, give identical `stored_rows` outputs for the same steps, while
    /// their slots hold different rows: each step's launch rows are the rows
    /// it names, a padding row past its batch slots and a failed selection
    /// (both -1), which read zeros, not whatever row slot 0 holds.
    #[test]
    fn a_cold_and_a_warm_table_read_identical_rows() {
        use magnitude_kernels::stored_rows;
        let device = device();
        let kernel = stored_rows::native_for_device_with(
            &device,
            stored_rows::Elements {
                E: resident(),
                A: Element::bf16(),
            },
            &seismic::generated::native_implementation::<stored_rows::Entry>(&device)
                .unwrap()
                .unwrap()
                .default_specialization(&seismic::NativeSpecialization::new().with_static("W", WIDTH))
                .unwrap(),
        )
        .unwrap();
        let steps = [[30, 41], [2, 3], [5, 7], [30, 2], [9, 41], [3, 5]];
        for form in [DirectoryForm::Dense, DirectoryForm::Hashed] {
            let kept = std::env::temp_dir().join(format!(
                "magnitude-stored-table-kept-{form:?}-{}",
                std::process::id()
            ));
            let path = file(&format!("cold-warm-{form:?}"));
            // Each step's outputs, and the row slot 0 held when it read.
            let run = || {
                let table = StoredTable::new(
                    &device,
                    &descriptor(),
                    stored(),
                    resident(),
                    ROWS,
                    WIDTH,
                    StoredSlots::Whole { records: 1, record_rows: 4 },
                    form,
                    ranges(&path),
                    Some(KeptRows::of(&kept, "table")),
                )
                .unwrap();
                table.seed().unwrap();
                let read = steps
                    .iter()
                    .map(|&[first, second]| {
                        let step = table
                            .queue(StepRows::Known(vec![vec![first], vec![second]]))
                            .unwrap();
                        assert!(step.gate.wait_signalled(step.value, Duration::from_secs(10)).unwrap());
                        let selections = [first as i32, 0, -1, 0, second as i32, 0, -1, 1];
                        let tokens = seismic::Tensor::from_host(
                            &device,
                            Element::i32(),
                            &[4, 2],
                            &selections.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
                        )
                        .unwrap();
                        let rows = kernel
                            .call(stored_rows::Args {
                                slots: table.slots_tensor(),
                                tokens: &tokens,
                                directory: table.directory(),
                            })
                            .unwrap()
                            .value
                            .read_to_host()
                            .unwrap();
                        let slot_zero = table.shared.state.lock().unwrap().directory.held(0);
                        read_done(&step);
                        (rows, slot_zero)
                    })
                    .collect::<Vec<_>>();
                drop(table);
                read
            };
            let cold = run();
            let warm = run();
            // The loads differ where the -1 rows used to read.
            assert!(
                cold.iter().zip(&warm).any(|(cold, warm)| cold.1 != warm.1),
                "{form:?}: slot 0 held the same rows cold and warm"
            );
            // Bf16 rows of WIDTH: the padding row and the failed selection
            // read zeros.
            let row_bytes = 2 * WIDTH as usize;
            for ((cold, _), (warm, _)) in cold.iter().zip(&warm) {
                assert_eq!(cold, warm, "{form:?}: cold and warm outputs");
                for unnamed in [1, 3] {
                    assert!(cold[unnamed * row_bytes..][..row_bytes].iter().all(|byte| *byte == 0));
                }
            }
            std::fs::remove_file(path).unwrap();
            std::fs::remove_dir_all(kept).unwrap();
        }
    }
}
