//! Stored tables: a streamed gathered table whose slots hold rows as stored
//! (`TableFill::Stored`, an n-gram table), written by the host while the
//! steps that read them wait at a gate.
//!
//! The slots and the table's directory live in host-shared storage
//! (`seismic::Tensor::shared`), which the host writes in place while device
//! work is queued. Every step that reads the table carries two timeline
//! nodes (`stored_table_rows`): a gate before its read, and a signal after
//! it. Steps are numbered in queue order, and step `n` binds both at `n`:
//!
//! - the filler (a thread of the table's own) takes each step's rows in
//!   queue order, as the host knows them or once it learns them (a step
//!   whose tokens the device selected), writes every row the step names
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
//! usual. Rows are read without filling the file cache (`UncachedSource`).

use crate::device_table::KeptRows;
use crate::planning::DirectoryForm;
use crate::slot_directory::SlotDirectory;
use magnitude_artifacts::UncachedSource;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

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
    /// A notice for steps of at most `rows` rows on `device`.
    pub fn new(device: &seismic::Device, rows: u64) -> Result<Self, String> {
        let failed = |error: &dyn std::fmt::Display| format!("selection notice: {error}");
        Ok(Self {
            copies: seismic::Tensor::shared(device, seismic::Element::i32(), &[2, rows, 2])
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
        if !self.timeline.wait(self.value, SLOT_WAIT) {
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

/// A table whose slots the host fills as stored.
pub struct StoredTable {
    shared: Arc<Shared>,
    jobs: Mutex<Option<mpsc::Sender<Job>>>,
    filler: Option<JoinHandle<()>>,
    /// The last step queued.
    queued: Mutex<u64>,
}

struct Job {
    step: u64,
    rows: StepRows,
}

struct Shared {
    /// `[slots, width]` rows as stored.
    slots: seismic::Tensor,
    /// `[entries, fields]` I32 (`DirectoryForm`).
    directory: seismic::Tensor,
    fields: u64,
    row_bytes: usize,
    /// The table's rows in the artifact file: `rows` of `row_bytes` from
    /// `offset`.
    source: UncachedSource,
    offset: u64,
    rows: u64,
    gate: seismic::Timeline,
    read: seismic::Timeline,
    state: Mutex<State>,
    failures: Mutex<BTreeMap<u64, StepFailure>>,
    kept: Option<KeptRows>,
}

struct State {
    directory: SlotDirectory,
    /// Per slot, the latest step that named its row; 0 for none.
    named: Vec<u64>,
    /// Per slot, whether a step named its row, here or when the table was
    /// last served.
    served: Vec<bool>,
}

impl StoredTable {
    /// A table of `rows` rows of `width` values stored as `element` (each
    /// `row_bytes` bytes, from `offset` in the file `source` reads), in
    /// `slots` slots and a directory of `form`, on `device`. Its slots are
    /// empty (`seed`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &seismic::Device,
        element: seismic::Element,
        rows: u64,
        width: u64,
        slots: u64,
        form: DirectoryForm,
        source: UncachedSource,
        offset: u64,
        kept: Option<KeptRows>,
    ) -> Result<Self, String> {
        let failed = |error: &dyn std::fmt::Display| format!("stored table: {error}");
        let row_bytes = element
            .canonical_byte_len(&[1, width])
            .map_err(|error| failed(&error))?;
        if source.size() < offset + rows * row_bytes {
            return Err(failed(&"the artifact does not hold the table's rows"));
        }
        if slots == 0 || slots > rows || slots > i32::MAX as u64 || rows > i32::MAX as u64 {
            return Err(failed(&format!("{slots} slots for {rows} rows")));
        }
        let directory = SlotDirectory::new(rows, slots, 1, form);
        let (entries, fields) = (form.entries(rows, slots), form.fields());
        let slots_tensor = seismic::Tensor::shared(device, element, &[slots, width])
            .map_err(|error| failed(&error))?;
        let directory_tensor =
            seismic::Tensor::shared(device, seismic::Element::i32(), &[entries, fields])
                .map_err(|error| failed(&error))?;
        let shared = Arc::new(Shared {
            slots: slots_tensor,
            directory: directory_tensor,
            fields,
            row_bytes: row_bytes as usize,
            source,
            offset,
            rows,
            gate: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            read: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            state: Mutex::new(State {
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
        let rows = self
            .shared
            .kept
            .as_ref()
            .map_or_else(Vec::new, KeptRows::read)
            .into_iter()
            .filter(|row| *row < self.shared.rows)
            .take(self.shared.slots.extents()[0] as usize)
            .collect::<Vec<_>>();
        let mut state = self.shared.state.lock().expect("stored table lock");
        match self.shared.bring(&mut state, 0, &[rows], true) {
            Ok(None) => Ok(()),
            Ok(Some(failure)) => Err(failure.error),
            Err(error) => Err(error),
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
        let taken = self.shared.gate.wait(value, SLOT_WAIT)
            && self.shared.read.wait(value.saturating_sub(1), SLOT_WAIT);
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
        let state = self.shared.state.lock().expect("stored table lock");
        self.shared.keep(&state);
    }
}

impl Shared {
    /// The filler: each step in queue order.
    fn fill_steps(&self, jobs: mpsc::Receiver<Job>) {
        for Job { step, rows } in jobs {
            let failure = match rows {
                StepRows::Known(rows) => Ok(rows),
                StepRows::Selected(rows) => rows(),
            }
            .and_then(|rows| {
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
        if naming.len() > state.named.len() {
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
                // read it: this step, or one whose read is not done.
                let kept = |slot: usize| {
                    claimed.contains(&slot)
                        || (step != 0 && named[slot] == step)
                        || !read.reached(named[slot])
                };
                if let Some((slot, _)) = state.directory.free(kept) {
                    break slot;
                }
                // Every slot holds a row a step still reads: wait for the
                // oldest such read.
                let oldest = state
                    .named
                    .iter()
                    .copied()
                    .filter(|named| *named != step)
                    .min()
                    .ok_or("a step names more rows than the table has slots")?;
                if !self.read.wait(oldest, SLOT_WAIT) {
                    return Err("no slot of the stored table came free".into());
                }
            };
            state.named[slot] = step;
            claimed.insert(slot);
            taken.push(slot);
        }
        // Read the absent rows straight into their slots, then publish
        // those that were read; a row that was not leaves its slot free.
        let ranges = absent
            .iter()
            .map(|row| (self.offset + row * self.row_bytes as u64, self.row_bytes))
            .collect::<Vec<_>>();
        let mut read = vec![false; absent.len()];
        self.source.read_ranges(&ranges, |index, bytes| {
            let written = bytes
                .map_err(|error| error.to_string())
                .and_then(|bytes| {
                    let offset = (taken[index] * self.row_bytes) as u64;
                    // SAFETY: the slot's previous row has been read by every
                    // step that named it, and no queued step names the
                    // slot's new row before its gate opens.
                    unsafe { self.slots.write_shared(offset, bytes) }
                        .map_err(|error| error.to_string())
                });
            match written {
                Ok(()) => read[index] = true,
                Err(error) => {
                    failed.extend(naming[&absent[index]].iter().copied());
                    errors.push(format!("table row {}: {error}", absent[index]));
                }
            }
        });
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
            if step > 1 && !self.read.wait(step - 1, SLOT_WAIT) {
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
    use seismic::{BackendName, DeviceCatalog, Element};

    const ROWS: u64 = 4096;
    const WIDTH: u64 = 160;
    const ROW_BYTES: usize = 90;

    /// Table row `row` as stored: five IQ4_NL blocks whose scale bytes name
    /// the row and whose nibbles follow it.
    fn row(row: u64) -> Vec<u8> {
        (0..ROW_BYTES)
            .map(|byte| (row as usize * 7 + byte * 13) as u8)
            .collect()
    }

    fn table(name: &str, slots: u64, form: DirectoryForm) -> (std::path::PathBuf, StoredTable) {
        let path = std::env::temp_dir().join(format!(
            "magnitude-stored-table-{name}-{}.bin",
            std::process::id()
        ));
        let mut bytes = vec![0xee; 100];
        bytes.extend((0..ROWS).flat_map(row));
        std::fs::write(&path, &bytes).unwrap();
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        let table = StoredTable::new(
            &device,
            Element::named("gguf_iq4_nl_block").unwrap(),
            ROWS,
            WIDTH,
            slots,
            form,
            UncachedSource::open(&path).unwrap(),
            100,
            None,
        )
        .unwrap();
        (path, table)
    }

    /// The stored row a reader finds for `row`, probing the device's
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
        let mut bytes = vec![0u8; ROW_BYTES];
        // SAFETY: as above.
        unsafe { table.slots_tensor().read_shared((slot * ROW_BYTES) as u64, &mut bytes) }
            .unwrap();
        Some(bytes)
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
                assert!(queued.gate.wait(queued.value, Duration::from_secs(10)));
                assert_eq!(table.failure(queued.value), None);
                for row in &rows {
                    assert_eq!(found(&table, *row), Some(self::row(*row)), "{form:?} row {row}");
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
        assert!(first.gate.wait(1, Duration::from_secs(10)));
        // The next step needs a slot while every slot holds a row the first
        // step has not read: its gate waits for that read.
        let second = table.queue(StepRows::Known(vec![vec![5]])).unwrap();
        assert!(!second.gate.wait(2, Duration::from_millis(100)));
        for row in [1, 2, 3, 4] {
            assert_eq!(found(&table, row), Some(self::row(row)));
        }
        read_done(&first);
        assert!(second.gate.wait(2, Duration::from_secs(10)));
        assert_eq!(found(&table, 5), Some(row(5)));
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
        assert!(!step.gate.reached(1));
        sender.send(vec![vec![7], vec![9]]).unwrap();
        assert!(step.gate.wait(1, Duration::from_secs(10)));
        assert_eq!(found(&table, 9), Some(row(9)));
        read_done(&step);
        // A row outside the table fails the slot that named it, and the
        // step's gate opens.
        let failed = table.queue(StepRows::Known(vec![vec![2], vec![ROWS]])).unwrap();
        assert!(failed.gate.wait(2, Duration::from_secs(10)));
        let failure = table.failure(2).unwrap();
        assert_eq!(failure.slots, FailedSlots::Only(BTreeSet::from([1])));
        read_done(&failed);
        // Rows that never reach the host fail every slot.
        let lost = table
            .queue(StepRows::Selected(Box::new(|| Err("never selected".into()))))
            .unwrap();
        assert!(lost.gate.wait(3, Duration::from_secs(10)));
        assert_eq!(table.failure(3).unwrap().slots, FailedSlots::Every);
        read_done(&lost);
        // A failure stays until its step is retired.
        let later = table.queue(StepRows::Known(vec![vec![1]])).unwrap();
        assert!(later.gate.wait(4, Duration::from_secs(10)));
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
        assert!(step.gate.wait(1, Duration::from_secs(10)));
        let failure = table.failure(1).unwrap();
        assert_eq!(failure.slots, FailedSlots::Only(BTreeSet::from([1])));
        assert!(failure.fails(1) && !failure.fails(0));
        for row in [1, 2, 3] {
            assert_eq!(found(&table, row), Some(self::row(row)));
        }
        assert_eq!(found(&table, 3500), None);
        read_done(&step);
        // The next step reads as usual.
        let next = table.queue(StepRows::Known(vec![vec![4], vec![5]])).unwrap();
        assert!(next.gate.wait(2, Duration::from_secs(10)));
        assert_eq!(table.failure(2), None);
        assert_eq!(found(&table, 5), Some(row(5)));
        read_done(&next);
        drop(table);
        std::fs::remove_file(path).unwrap();
    }
}
