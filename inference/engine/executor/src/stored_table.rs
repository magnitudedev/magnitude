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

use crate::device_table::KeptRows;
use crate::planning::DirectoryForm;
use crate::slot_directory::SlotDirectory;
use seismic_io::{Reader, Ended, Priority, Range};
use seismic::cpu::repack::RowConversion;
use seismic::Element;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

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
    pub gate: seismic::Timeline,
    pub read: seismic::Timeline,
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
    timeline: seismic::Timeline,
    /// The last step's value.
    next: Mutex<u64>,
}

/// One step's selection notice: the copy it writes and its value.
pub struct NoticeStep {
    pub tokens: seismic::Tensor,
    pub timeline: seismic::Timeline,
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
    jobs: Mutex<Option<mpsc::Sender<Job>>>,
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
    fn new(records: u64, record_rows: u64, row_bytes: u64) -> Self {
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
                row_bytes,
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
    /// How a row as stored becomes a slot's row.
    conversion: &'static RowConversion,
    width: u64,
    /// The bytes of a row as stored and of a slot's row.
    row_bytes: usize,
    slot_bytes: usize,
    /// The table's rows in the artifact file: `rows` of `row_bytes` from
    /// `offset`.
    source: Reader,
    offset: u64,
    rows: u64,
    gate: seismic::Timeline,
    read: seismic::Timeline,
    state: Mutex<State>,
    failures: Mutex<BTreeMap<u64, StepFailure>>,
    kept: Option<KeptRows>,
}

struct State {
    /// The leading records whose slots hold rows: every one committed.
    records: u64,
    directory: SlotDirectory,
    /// Per slot, the latest step that named its row; 0 for none.
    named: Vec<u64>,
    /// Per slot, whether a step named its row, here or when the table was
    /// last served.
    served: Vec<bool>,
}

impl StoredTable {
    /// A table of `rows` rows of `width` values stored as `stored` (from
    /// `offset` in the file `source` reads), in `layout`'s slots of the
    /// `resident` representation, every record committed, and a directory
    /// of `form`, on `device`. Its slots are empty (`seed`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &seismic::Device,
        stored: seismic::Element,
        resident: seismic::Element,
        rows: u64,
        width: u64,
        layout: StoredSlots,
        form: DirectoryForm,
        source: Reader,
        offset: u64,
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
        let piece = (conversion.source_row_bytes(values), conversion.stored_row_bytes(values));
        if piece != (row_bytes as usize, Some(slot_bytes as usize)) {
            return Err(failed(&"a stored row does not convert into one slot's row"));
        }
        if source.size() < offset + rows * row_bytes {
            return Err(failed(&"the artifact does not hold the table's rows"));
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
        let shared = Arc::new(Shared {
            slots: slots_tensor,
            placement,
            record_rows,
            stack: Mutex::new(RowStack::new(records, record_rows, row_bytes)),
            directory: directory_tensor,
            fields,
            conversion,
            width,
            row_bytes: row_bytes as usize,
            slot_bytes: slot_bytes as usize,
            source,
            offset,
            rows,
            gate: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            read: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            state: Mutex::new(State {
                records,
                directory,
                named: vec![0; slots as usize],
                served: vec![false; slots as usize],
            }),
            failures: Mutex::new(BTreeMap::new()),
            kept,
        });
        // The directory starts with every entry empty.
        {
            let mut state = shared.state.lock().expect("stored table lock");
            let bytes = state.directory.entry_bytes();
            // SAFETY: no device work uses the table yet.
            unsafe { shared.directory.write_shared(0, &bytes) }.map_err(|error| failed(&error))?;
        }
        let (jobs, receiver) = mpsc::channel::<Job>();
        let filler = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("stored-table-filler".into())
                .spawn(move || shared.fill_steps(receiver))
                .map_err(|error| failed(&error))?
        };
        Ok(Self {
            shared,
            jobs: Mutex::new(Some(jobs)),
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
            Ok(None) => Ok(()),
            Ok(Some(failure)) => Err(failure.error),
            Err(error) => Err(error),
        }
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
            state.named[held..].fill(0);
            state.served[held..].fill(false);
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
        self.jobs
            .lock()
            .expect("stored table jobs lock")
            .as_ref()
            .ok_or("the stored table's filler has stopped")?
            .send(Job { step, rows })
            .map_err(|_| "the stored table's filler has stopped".to_owned())?;
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
        // The filler opens the gate of every step it takes; the read
        // timeline says every step up to a value has read.
        let lost = |error: seismic::ExecutionError| format!("an abandoned stored-table step's predecessors: {error}");
        let taken = self.shared.gate.wait(value, SLOT_WAIT).map_err(lost)?
            && self.shared.read.wait(value.saturating_sub(1), SLOT_WAIT).map_err(lost)?;
        if !taken {
            return Err("an abandoned stored-table step's predecessors did not read".into());
        }
        self.shared
            .read
            .signal(value)
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
        drop(self.jobs.lock().expect("stored table jobs lock").take());
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
    /// The filler: each step in queue order.
    fn fill_steps(&self, jobs: mpsc::Receiver<Job>) {
        for Job { step, rows } in jobs {
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
                let failure = self.bring(&mut state, step, &rows, true)?;
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
        }
    }

    /// Hold every row `rows` names per batch slot for step `step` (0 before
    /// the first step): rows a slot holds are named; the others are read
    /// into free slots and published. `served`: steps named the rows. The
    /// rows that could not be read are not published, and fail the batch
    /// slots that named them (`Ok(Some)`); a failure that is no row's fails
    /// the step (`Err`).
    fn bring(
        &self,
        state: &mut State,
        step: u64,
        rows: &[Vec<u64>],
        served: bool,
    ) -> Result<Option<StepFailure>, String> {
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
                    state.named[slot as usize] = step;
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
                    Ok(claimed.contains(&slot) || (step != 0 && named[slot] == step) || !read.reached(named[slot])?)
                };
                let free = state.directory.try_free(kept).map_err(|error: seismic::ExecutionError| format!("the stored table's reads: {error}"))?;
                if let Some((slot, _)) = free {
                    break slot;
                }
                // Every slot holds a row a step still reads: wait for the
                // oldest such read.
                let oldest = state.named[..held]
                    .iter()
                    .copied()
                    .filter(|named| *named != step)
                    .min()
                    .ok_or("a step names more rows than the table has slots")?;
                if !self.read.wait(oldest, SLOT_WAIT).map_err(|error| format!("the stored table's reads: {error}"))? {
                    return Err("no slot of the stored table came free".into());
                }
            };
            state.named[slot] = step;
            claimed.insert(slot);
            taken.push(slot);
        }
        // Read the absent rows as stored, convert each into its slot, then
        // publish those that were read; a row that was not leaves its slot
        // free.
        let stored = Arc::new(seismic_io::Buffer::new(absent.len() * self.row_bytes));
        let ranges = absent
            .iter()
            .enumerate()
            .map(|(index, row)| Range {
                offset: self.offset + row * self.row_bytes as u64,
                length: self.row_bytes,
                into: stored.clone(),
                at: index * self.row_bytes,
            })
            .collect();
        // SAFETY: the ranges are distinct rows of `stored`, which nothing
        // reads until the job has ended.
        let ended = unsafe { self.source.submit(ranges, Priority::Demand) }
            .map_err(|error| format!("stored table rows: {error}"))?
            .wait();
        let unread = match ended {
            Ended::Read { failed } => failed,
            Ended::Panicked(panic) => panic.resume(),
        };
        let mut read = vec![true; absent.len()];
        for (index, error) in unread {
            read[index] = false;
            failed.extend(naming[&absent[index]].iter().copied());
            errors.push(format!("table row {}: {error}", absent[index]));
        }
        let mut converted = vec![0u8; self.slot_bytes];
        // SAFETY: the job that read into `stored` has ended.
        let stored = unsafe { stored.bytes() };
        for (index, row) in stored.chunks_exact(self.row_bytes).enumerate() {
            if !read[index] {
                continue;
            }
            self.convert(row, &mut converted);
            let offset = self.slot_offset(taken[index] as u64);
            // SAFETY: the slot's previous row has been read by every step
            // that named it, and no queued step names the slot's new row
            // before its gate opens.
            if let Err(error) = unsafe { self.slots.write_shared(offset, &converted) } {
                read[index] = false;
                failed.extend(naming[&absent[index]].iter().copied());
                errors.push(format!("table row {}: {error}", absent[index]));
            }
        }
        for ((row, slot), read) in absent.iter().zip(&taken).zip(read) {
            if read {
                state.directory.hold(*row, *slot, true);
                state.served[*slot] = served;
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
        Ok((!failed.is_empty()).then(|| StepFailure {
            slots: FailedSlots::Only(failed),
            error: format!("stored table rows: {}", errors.join("; ")),
        }))
    }

    /// Where slot `slot` lies in the slots' storage: its row of its record,
    /// where the storage lays the record (`Tensor::record_placement`).
    fn slot_offset(&self, slot: u64) -> u64 {
        self.placement.offset(slot / self.record_rows) + (slot % self.record_rows) * self.slot_bytes as u64
    }

    /// The slot row of the row `stored` as stored (`row_conversion`).
    fn convert(&self, stored: &[u8], slot: &mut [u8]) {
        // `stored` is a read range of `row_bytes`, one row of `width` values
        // as stored, and `slot` a buffer of `slot_bytes`, one resident row:
        // one piece of the conversion (`Shared::new`).
        self.conversion
            .piece(self.width as usize, stored, slot)
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
        resident().repack_source(stored(), bytes).unwrap()
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
            stored(),
            resident(),
            ROWS,
            WIDTH,
            StoredSlots::Whole { records: 1, record_rows: slots },
            form,
            Reader::open(&path, CachePolicy::Uncached).unwrap(),
            100,
            None,
        )
        .unwrap();
        (path, table)
    }

    fn device() -> seismic::Device {
        seismic::Devices::discover(&[])
            .unwrap()
            .open_backend(BackendName::Cpu)
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
        let table = StoredTable::new(
            &device(),
            stored(),
            resident(),
            ROWS,
            WIDTH,
            StoredSlots::Reserved { records: 4, record_rows: 256 },
            DirectoryForm::Hashed,
            Reader::open(&path, CachePolicy::Uncached).unwrap(),
            100,
            None,
        )
        .unwrap();
        let rows = |count: u64, from: u64| (from..from + count).map(|row| row * 3 % ROWS).collect::<Vec<_>>();
        let run = |rows: Vec<u64>| {
            let step = table.queue(StepRows::Known(vec![rows])).unwrap();
            assert!(step.gate.wait(step.value, Duration::from_secs(10)).unwrap());
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
        assert!(table.charged_bytes().unwrap() < before);
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

    /// The row stack counts each decode step's rows by their depth in
    /// records: every row of a record of no more rows than it samples is
    /// sampled, and a row read `k` distinct rows ago is `k / 256` records
    /// deep; a row never read is deeper than every record. A step's cost
    /// falls as records hold more of its rows.
    #[test]
    fn the_row_stack_counts_decode_rows_by_their_depth_in_records() {
        let mut stack = RowStack::new(4, SAMPLED_PER_RECORD, 90);
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
        step.read.signal(step.value).unwrap();
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
                assert!(queued.gate.wait(queued.value, Duration::from_secs(10)).unwrap());
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
        assert!(first.gate.wait(1, Duration::from_secs(10)).unwrap());
        // The next step needs a slot while every slot holds a row the first
        // step has not read: its gate waits for that read.
        let second = table.queue(StepRows::Known(vec![vec![5]])).unwrap();
        assert!(!second.gate.wait(2, Duration::from_millis(100)).unwrap());
        for row in [1, 2, 3, 4] {
            assert_eq!(found(&table, row), Some(slot_row(row)));
        }
        read_done(&first);
        assert!(second.gate.wait(2, Duration::from_secs(10)).unwrap());
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

    #[test]
    fn rows_learned_late_and_failures_still_open_the_gate() {
        let (path, table) = table("selected", 8, DirectoryForm::Dense);
        let (sender, receiver) = mpsc::channel::<Vec<Vec<u64>>>();
        let step = table
            .queue(StepRows::Selected(Box::new(move || {
                receiver.recv().map_err(|error| error.to_string())
            })))
            .unwrap();
        assert!(!step.gate.reached(1).unwrap());
        sender.send(vec![vec![7], vec![9]]).unwrap();
        assert!(step.gate.wait(1, Duration::from_secs(10)).unwrap());
        assert_eq!(found(&table, 9), Some(slot_row(9)));
        read_done(&step);
        // A row outside the table fails the slot that named it, and the
        // step's gate opens.
        let failed = table.queue(StepRows::Known(vec![vec![2], vec![ROWS]])).unwrap();
        assert!(failed.gate.wait(2, Duration::from_secs(10)).unwrap());
        let failure = table.failure(2).unwrap();
        assert_eq!(failure.slots, FailedSlots::Only(BTreeSet::from([1])));
        read_done(&failed);
        // Rows that never reach the host fail every slot.
        let lost = table
            .queue(StepRows::Selected(Box::new(|| Err("never selected".into()))))
            .unwrap();
        assert!(lost.gate.wait(3, Duration::from_secs(10)).unwrap());
        assert_eq!(table.failure(3).unwrap().slots, FailedSlots::Every);
        read_done(&lost);
        // A failure stays until its step is retired.
        let later = table.queue(StepRows::Known(vec![vec![1]])).unwrap();
        assert!(later.gate.wait(4, Duration::from_secs(10)).unwrap());
        assert_eq!(table.failure(4), None);
        assert!(table.failure(2).is_some());
        table.retire(3);
        assert_eq!(table.failure(2), None);
        assert_eq!(table.failure(3), None);
        read_done(&later);
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
        assert!(step.gate.wait(1, Duration::from_secs(10)).unwrap());
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
        assert!(next.gate.wait(2, Duration::from_secs(10)).unwrap());
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
                    stored(),
                    resident(),
                    ROWS,
                    WIDTH,
                    StoredSlots::Whole { records: 1, record_rows: 4 },
                    form,
                    Reader::open(&path, CachePolicy::Uncached).unwrap(),
                    100,
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
                        assert!(step.gate.wait(step.value, Duration::from_secs(10)).unwrap());
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
