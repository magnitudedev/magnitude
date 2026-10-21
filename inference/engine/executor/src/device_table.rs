//! Gathered tables on the device.
//!
//! A gathered table is a weight a kernel reads by row index. The device holds
//! it whole, or streamed (`DeviceTable`): slots for some of its rows and a
//! directory of every row's slot, with rows reaching their slots from the
//! artifact as steps name them. Which of the two is the weight plan's
//! (`WeightPlan::streamed`); a reader holds a `ResidentTable` and reads
//! either the same way.
//!
//! Nothing here knows which table it holds or what its readers compute.

use crate::planning::DirectoryForm;
use crate::slot_directory::SlotDirectory;
use crate::stored_table::StoredTable;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

/// How long the filler waits for the device (a read's notice, the read
/// before it): far beyond any step's device time.
const DEVICE_WAIT: Duration = Duration::from_secs(60);

/// How often the filler, waiting for a read's notice, looks whether the read
/// was abandoned.
const ABANDON_POLL: Duration = Duration::from_millis(10);

/// A gathered table as its readers find it on the device.
///
/// A reader whose rows the host knows gathers by `indices`. A reader whose
/// rows the device selects finds their slots in the directory, behind its
/// table's gate (`DeviceTable::read_selected`): every row it selects is
/// found.
///
/// A stored table (`StoredTable`) is read only through its directory, by
/// steps that wait at its gate until the host has written the rows they
/// name: every row such a read names is found.
#[derive(Clone)]
pub enum ResidentTable {
    /// A table the device holds whole: a row is its own index.
    Whole,
    Streamed(Rc<DeviceTable>),
    Stored(Arc<StoredTable>),
}

impl ResidentTable {
    /// Visit the tensors the table holds beside its weight: a streamed
    /// table's directory, staging, landing staging and notice, a stored
    /// table's directory.
    pub(crate) fn visit_extra_storage(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        match self {
            Self::Whole => {}
            Self::Streamed(table) => {
                visit(&table.shared.directory);
                visit(&table.staging);
                visit(&table.shared.landing_staging);
                visit(&table.shared.notice);
            }
            Self::Stored(table) => visit(table.directory()),
        }
    }

    /// A table the device holds whole: a row is its own index.
    pub fn whole() -> Self {
        Self::Whole
    }

    pub fn streamed(table: Rc<DeviceTable>) -> Self {
        Self::Streamed(table)
    }

    /// A streamed or stored table's directory (`gathered_row`).
    pub fn directory(&self) -> Option<&seismic::Tensor> {
        match self {
            Self::Whole => None,
            Self::Streamed(table) => Some(&table.shared.directory),
            Self::Stored(table) => Some(table.directory()),
        }
    }

    /// The stored table this is.
    pub fn stored(&self) -> Option<&Arc<StoredTable>> {
        match self {
            Self::Stored(table) => Some(table),
            _ => None,
        }
    }

    /// The streamed table this is.
    pub fn device(&self) -> Option<&Rc<DeviceTable>> {
        match self {
            Self::Streamed(table) => Some(table),
            _ => None,
        }
    }

    /// What a streamed table allocates beside its slots: its directory, its
    /// staging, its landing staging and its notice
    /// (`WeightPlan::streamed_bytes`).
    pub fn streamed_bytes(&self) -> u64 {
        match self {
            Self::Whole => 0,
            Self::Streamed(table) => {
                table.shared.directory.storage_bytes()
                    + table.staging.storage_bytes()
                    + table.shared.landing_staging.storage_bytes()
                    + table.shared.notice.storage_bytes()
            }
            Self::Stored(table) => table.directory_bytes(),
        }
    }

    /// The indices a kernel gathers `rows` by: the rows themselves, or a
    /// streamed table's slots, which brings the rows the device lacks ahead
    /// of the work queued next. A stored table's rows are read only through
    /// its directory.
    pub fn indices(&self, rows: &[i32]) -> Result<Vec<i32>, String> {
        let table = match self {
            Self::Whole => return Ok(rows.to_vec()),
            Self::Streamed(table) => table,
            Self::Stored(_) => {
                return Err("a stored table's rows are read through its directory".into());
            }
        };
        let rows = rows
            .iter()
            .map(|row| u64::try_from(*row).map_err(|_| format!("{row} names no table row")))
            .collect::<Result<Vec<_>, _>>()?;
        table.slots(&rows)
    }
}

/// Tiles bound for adjacent slots: the first rows of the tiles, in the order
/// of their slots, and the tensor of those slots.
pub struct TileRun<'a> {
    pub tiles: &'a [u64],
    pub slots: seismic::Tensor,
}

/// Where a streamed table's tiles come from.
pub trait TileSource {
    /// Queue the tiles of every run into their slots, together, through
    /// `staging`: the tiles' rows as the table stores them, run after run.
    /// Device work queued after it finds the tiles in their slots.
    fn bring(
        &self,
        runs: &[TileRun<'_>],
        staging: &mut seismic::Tensor,
    ) -> Result<TilesBrought, String>;

    /// The table's rows as stored, which the filler lands for the reads of
    /// rows the device selects.
    fn rows(&self) -> Arc<dyn StoredRows>;
}

/// A streamed table's rows as the artifact stores them.
pub trait StoredRows: Send + Sync {
    /// The bytes of the rows from `first`, as many as `out` holds whole.
    fn read(&self, first: u64, out: &mut [u8]) -> Result<(), String>;
}

/// Waits for queued tiles to reach their slots, after which their staging
/// holds nothing the device reads.
pub type TilesBrought = Box<dyn FnOnce() -> Result<(), String>>;

/// A streamed gathered table: the device holds slots for only some of the
/// table's row tiles (a tile is the rows its storage lays out together; one
/// row in `rows16`), landing slots after them, and a directory of every
/// row's slot.
///
/// A read of rows the host knows is given their slots (`indices`): absent
/// tiles are brought into slots no step named since the clock last passed
/// them, by device work queued ahead of the read.
///
/// A read of rows the device selects (`read_selected`) waits at the table's
/// gate. Ahead of the gate its step copies the selection for the host (its
/// notice). The table's filler thread takes the reads in queue order; for
/// each, once the read before it has read, it lands every selected tile the
/// slots do not hold, as stored, into the landing staging, points its
/// directory entry at its landing slot and opens the gate. Behind the gate
/// the step converts the landing staging into the landing slots with the
/// table weight's own import entry, then reads. Once the step completed,
/// the tiles it landed are brought into slots of their own
/// (`SubmittedRead::complete`). No such read finds another row.
///
/// The directory lives in host-shared storage, and only the filler writes
/// it: a read's entries are published before its gate opens, while no other
/// read is between its gate and its read signal. A tile the owner brings is
/// pending until the first read queued after its conversion, the earliest
/// to read it there; a tile the owner replaces leaves when the next read is
/// filled, and its slot is overwritten only by device work queued after
/// every read the filler pointed at it. So every read finds each row it
/// names in a slot that holds it when it reads. The owner never waits for
/// the device but to reuse its staging.
pub struct DeviceTable {
    /// The slots, then the landing slots: the leading rows of the table's
    /// shape.
    slots: seismic::Tensor,
    /// The rows a host step's tiles pass through, as stored.
    staging: seismic::Tensor,
    source: Box<dyn TileSource>,
    kept: Option<KeptRows>,
    /// The tiles the last host step brought, while the device may still
    /// read their staging.
    brought: RefCell<Option<TilesBrought>>,
    shared: Arc<Shared>,
    /// The reads the filler takes: each one's value and rows.
    reads: Mutex<Option<mpsc::Sender<(u64, u64)>>>,
    filler: Option<JoinHandle<()>>,
}

/// What the owner and the filler share.
struct Shared {
    /// `[entries, fields]` I32 in host-shared storage (`DirectoryForm`).
    directory: seismic::Tensor,
    fields: u64,
    /// The landing staging: the rows of the landing slots' tiles, as stored,
    /// in host-shared storage.
    landing_staging: seismic::Tensor,
    /// A read's `[rows, 2]` (token, status) selection, which its step copies
    /// into host-shared storage.
    notice: seismic::Tensor,
    /// Raised by the device to a read's value once its notice is written.
    noticed: seismic::Timeline,
    /// Raised by the filler to a read's value once its tiles are landed.
    gate: seismic::Timeline,
    /// Raised by the device to a read's value once it has read.
    read: seismic::Timeline,
    rows: Arc<dyn StoredRows>,
    /// Bytes of one stored row.
    row_bytes: usize,
    /// Tiles of the regular slots; the landing slots' tiles follow them.
    slot_tiles: usize,
    landing_tiles: usize,
    state: Mutex<State>,
}

struct State {
    /// Every tile of slots, regular and landing.
    tiles: SlotDirectory,
    /// Per regular tile of slots, whether a step ever named its tile, here
    /// or when the table was last served: the tiles kept for the next time.
    served: Vec<bool>,
    /// Per regular tile of slots, the last host step that named its tile.
    named: Vec<u32>,
    /// The host steps queued so far.
    step: u32,
    /// The regular tiles of slots given a tile before the first step, the
    /// first ones.
    seeded: usize,
    /// The last selected read queued; reads are numbered from 1.
    queued: u64,
    /// The last selected read the filler took.
    filled: u64,
    /// Tiles brought into regular slots while reads queued before them were
    /// not yet filled, which no read finds yet.
    pending: Vec<Pending>,
    /// Per filled read, the first rows of the tiles its filler landed, or
    /// why its tiles could not be landed.
    landed: BTreeMap<u64, Result<Vec<u64>, String>>,
    /// Reads queued but never submitted, which the filler has not taken.
    abandoned: BTreeSet<u64>,
}

/// A tile brought into regular slots: reads from `from` on, which are
/// queued after its conversion, find it there. `used`: whether the clock
/// passes it once before it takes it.
#[derive(Clone, Copy)]
struct Pending {
    first: u64,
    slot: usize,
    from: u64,
    used: bool,
}

impl State {
    fn next_step(&mut self) {
        self.step = self.step.wrapping_add(1);
        if self.step == 0 {
            // The step count came round: no slot is named by a step yet.
            self.named.fill(0);
            self.step = 1;
        }
    }

    /// The regular tile of slots of the tile of first row `first`: held, or
    /// pending.
    fn regular(&self, first: u64, slot_tiles: usize) -> Option<usize> {
        let tile = self.tiles.tile_rows();
        self.tiles
            .slot(first)
            .map(|slot| (slot / tile) as usize)
            .filter(|slot| *slot < slot_tiles)
            .or_else(|| {
                self.pending
                    .iter()
                    .find(|pending| pending.first == first)
                    .map(|pending| pending.slot)
            })
    }
}

/// The rows steps named, kept in a file for the next time the table is
/// served. An optimization: a file that cannot be read or written keeps
/// none.
pub struct KeptRows(PathBuf);

impl KeptRows {
    /// The file in `directory` of the table `table` describes: everything
    /// its rows and their slots depend on.
    pub fn of(directory: &Path, table: &str) -> Self {
        let name = Sha256::digest(table.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Self(directory.join(format!("{name}.rows")))
    }

    pub(crate) fn read(&self) -> Vec<u64> {
        std::fs::read_to_string(&self.0)
            .ok()
            .and_then(|text| {
                text.split_whitespace()
                    .map(|row| row.parse().ok())
                    .collect::<Option<Vec<_>>>()
            })
            .unwrap_or_default()
    }

    pub(crate) fn write(&self, rows: &[u64]) {
        let text = rows.iter().map(u64::to_string).collect::<Vec<_>>().join(" ");
        let _ = self
            .0
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&self.0, text));
    }
}

/// A read of rows the device selects, queued but not yet submitted: what
/// its step binds. Dropped unsubmitted, it is abandoned: its filler opens its
/// gate and its read is done, so the reads after it are filled.
pub struct SelectedRead {
    table: Rc<DeviceTable>,
    /// The read's value on the table's timelines.
    pub value: u64,
    /// Where the step copies its `[rows, 2]` selection for the host.
    pub notice: seismic::Tensor,
    pub noticed: seismic::Timeline,
    pub gate: seismic::Timeline,
    pub read: seismic::Timeline,
    /// The landing staging of the read's tiles (`[rows, ..]` as stored),
    /// and the landing slots its step converts it into (`[rows, ..]`).
    pub staging: seismic::Tensor,
    pub landing: seismic::Tensor,
    submitted: bool,
}

impl SelectedRead {
    /// The read's step was submitted: the device raises its timelines.
    pub fn submitted(mut self) -> SubmittedRead {
        self.submitted = true;
        SubmittedRead {
            table: self.table.clone(),
            value: self.value,
        }
    }
}

impl Drop for SelectedRead {
    fn drop(&mut self) {
        if !self.submitted {
            self.table.abandon(self.value);
        }
    }
}

/// A submitted read of rows the device selects. Once its step completed,
/// `complete` brings the tiles it landed into slots of their own; dropped
/// instead, they are landed again by the next read that names them.
pub struct SubmittedRead {
    table: Rc<DeviceTable>,
    value: u64,
}

impl SubmittedRead {
    /// The read's step completed: the tiles it landed are brought into
    /// slots of their own, so that later reads find them there. Fails when
    /// its tiles could not be landed: the read found other rows, and its
    /// step's outputs are not to be used.
    pub fn complete(self) -> Result<(), String> {
        let landed = self.table.state().landed.remove(&self.value);
        match landed {
            Some(Ok(tiles)) => self.table.hold_landed(tiles),
            Some(Err(error)) => Err(error),
            None => Err(format!("selected read {} completed before it was filled", self.value)),
        }
    }
}

impl Drop for SubmittedRead {
    fn drop(&mut self) {
        self.table.state().landed.remove(&self.value);
    }
}

impl Drop for DeviceTable {
    /// The filler takes every read queued, so every gate opens, and stops.
    fn drop(&mut self) {
        drop(self.reads.lock().expect("device table reads lock").take());
        if let Some(filler) = self.filler.take() {
            let _ = filler.join();
        }
        let state = self.state();
        self.keep(&state);
    }
}

impl DeviceTable {
    /// The steps after which a table's rows are kept: a process that ends
    /// without dropping the table still leaves the rows its first steps
    /// named.
    const KEPT_AFTER_STEPS: u32 = 256;

    /// A table of `rows` rows, each of `element` and `row_shape` and stored
    /// as `stored`, in `slots` slots and `landing` landing slots on `device`,
    /// with staging for host steps that bring up to `staging_rows` rows. Its
    /// slots are empty (`seed`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &seismic::Device,
        element: seismic::Element,
        stored: seismic::Element,
        rows: u64,
        row_shape: &[u64],
        slots: u64,
        landing: u64,
        staging_rows: u64,
        form: DirectoryForm,
        source: Box<dyn TileSource>,
        kept: Option<KeptRows>,
    ) -> Result<Self, String> {
        let failed = |error: &dyn std::fmt::Display| format!("device table: {error}");
        let tile = element.tile_rows();
        if rows % tile != 0
            || slots % tile != 0
            || landing % tile != 0
            || staging_rows % tile != 0
            || slots == 0
        {
            return Err(format!(
                "a table of {rows} rows in {slots} slots and {landing} landing slots staged {staging_rows} rows at a time is not whole tiles of {tile} rows"
            ));
        }
        let shape = |rows: u64| {
            std::iter::once(rows)
                .chain(row_shape.iter().copied())
                .collect::<Vec<_>>()
        };
        let (slot_tiles, landing_tiles) = ((slots / tile) as usize, (landing / tile) as usize);
        // The directory addresses every slot; the clock takes regular ones
        // only.
        let mut tiles = SlotDirectory::new(rows, slots + landing, tile, form);
        let slots_tensor = seismic::Tensor::zeros(device, element, &shape(slots + landing))
            .map_err(|error| failed(&error))?;
        let directory = seismic::Tensor::shared(
            device,
            seismic::Element::i32(),
            &[form.entries(rows, slots + landing), form.fields()],
        )
        .map_err(|error| failed(&error))?;
        // SAFETY: no device work uses the table yet.
        unsafe { directory.write_shared(0, &tiles.entry_bytes()) }.map_err(|error| failed(&error))?;
        let shared = Arc::new(Shared {
            directory,
            fields: form.fields(),
            landing_staging: seismic::Tensor::shared(device, stored, &shape(landing))
                .map_err(|error| failed(&error))?,
            notice: seismic::Tensor::shared(device, seismic::Element::i32(), &[landing, 2])
                .map_err(|error| failed(&error))?,
            noticed: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            gate: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            read: seismic::Timeline::new(device).map_err(|error| failed(&error))?,
            rows: source.rows(),
            row_bytes: stored
                .canonical_byte_len(&shape(1))
                .map_err(|error| failed(&error))? as usize,
            slot_tiles,
            landing_tiles,
            state: Mutex::new(State {
                tiles,
                served: vec![false; slot_tiles],
                named: vec![0; slot_tiles],
                step: 0,
                seeded: 0,
                queued: 0,
                filled: 0,
                pending: Vec::new(),
                landed: BTreeMap::new(),
                abandoned: BTreeSet::new(),
            }),
        });
        let (reads, taken) = mpsc::channel();
        let filler = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("device-table-filler".into())
                .spawn(move || shared.fill_reads(taken))
                .map_err(|error| failed(&error))?
        };
        Ok(Self {
            staging: staging(&slots_tensor, stored, staging_rows)?,
            slots: slots_tensor,
            source,
            kept,
            brought: RefCell::new(None),
            shared,
            reads: Mutex::new(Some(reads)),
            filler: Some(filler),
        })
    }

    pub fn slots_tensor(&self) -> &seismic::Tensor {
        &self.slots
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.shared.state.lock().expect("device table lock")
    }

    fn tile_rows(&self) -> u64 {
        self.slots.element().tile_rows()
    }

    /// Fill the slots before the table's first step, the rows worth most
    /// first: the rows steps named when the table was last served, then the
    /// table's first rows. The clock reaches the last slots first, so the
    /// least valuable rows are the first replaced.
    pub fn seed(&self) -> Result<(), String> {
        let kept = self.kept.as_ref().map_or_else(Vec::new, KeptRows::read);
        self.hold(kept, true)?;
        self.hold(0..self.shared.slot_tiles as u64 * self.tile_rows(), false)
    }

    /// Bring the tiles of `rows` the device does not hold into the slots no
    /// tile was given yet, as many as those take, and wait for them: the
    /// table's first step finds the device idle. Rows outside the table name
    /// none. `served`: steps named these rows when the table was last
    /// served.
    fn hold(&self, rows: impl IntoIterator<Item = u64>, served: bool) -> Result<(), String> {
        let tile = self.tile_rows();
        let (absent, taken) = {
            let mut state = self.state();
            let mut absent = rows
                .into_iter()
                .filter(|row| state.tiles.has_row(*row))
                .map(|row| row - row % tile)
                .filter(|first| state.regular(*first, self.shared.slot_tiles).is_none())
                .collect::<Vec<_>>();
            absent.sort_unstable();
            absent.dedup();
            // Once a step named rows the slots are the steps'.
            let free = match state.step {
                0 => self.shared.slot_tiles - state.seeded,
                _ => 0,
            };
            absent.truncate(free);
            let taken = (state.seeded..state.seeded + absent.len()).collect::<Vec<_>>();
            state.seeded += absent.len();
            (absent, taken)
        };
        self.bring(&absent, &taken, served)?;
        let brought = self.brought.borrow_mut().take();
        brought.map_or(Ok(()), |brought| brought())
    }

    /// A read of rows the device selects, at most `rows` of them, is being
    /// queued: its bindings. The filler takes it once the device copied its
    /// selection.
    pub fn read_selected(self: &Rc<Self>, rows: u64) -> Result<SelectedRead, String> {
        let failed = |error: &dyn std::fmt::Display| format!("device table: {error}");
        let landing_rows = rows * self.tile_rows();
        if rows as usize > self.shared.landing_tiles {
            return Err(format!(
                "a read of {rows} selected rows exceeds the table's {} landing tiles",
                self.shared.landing_tiles
            ));
        }
        let landing_first = self.shared.slot_tiles as u64 * self.tile_rows();
        let (notice, staging, landing) = (
            self.shared.notice.slice_leading(0, rows),
            self.shared.landing_staging.slice_leading(0, landing_rows),
            self.slots
                .slice_leading(landing_first, landing_first + landing_rows),
        );
        let (notice, staging, landing) = (
            notice.map_err(|error| failed(&error))?,
            staging.map_err(|error| failed(&error))?,
            landing.map_err(|error| failed(&error))?,
        );
        let mut state = self.state();
        // A read is a step: the rows steps named are kept after the first
        // ones, whatever names them.
        state.next_step();
        if state.step == Self::KEPT_AFTER_STEPS {
            self.keep(&state);
        }
        let value = state.queued + 1;
        self.reads
            .lock()
            .expect("device table reads lock")
            .as_ref()
            .ok_or("the device table's filler has stopped")?
            .send((value, rows))
            .map_err(|_| "the device table's filler has stopped".to_owned())?;
        state.queued = value;
        Ok(SelectedRead {
            table: self.clone(),
            value,
            notice,
            noticed: self.shared.noticed.clone(),
            gate: self.shared.gate.clone(),
            read: self.shared.read.clone(),
            staging,
            landing,
            submitted: false,
        })
    }

    /// Read `value` was queued but will never be submitted: its filler
    /// opens its gate, and its read is done.
    fn abandon(&self, value: u64) {
        self.state().abandoned.insert(value);
    }

    /// The tiles a completed read landed are brought into slots of their
    /// own, as a step's absent tiles are, into slots no step named since
    /// the clock last passed them. A tile held or pending meanwhile is not
    /// brought again; one the clock finds no slot for is landed again when
    /// a read names it. No step is being queued: a step's tiles are replaced
    /// by device work queued after it.
    fn hold_landed(&self, tiles: Vec<u64>) -> Result<(), String> {
        let (absent, taken) = {
            let mut guard = self.state();
            let state = &mut *guard;
            let mut absent = tiles
                .into_iter()
                .filter(|first| state.regular(*first, self.shared.slot_tiles).is_none())
                .collect::<Vec<_>>();
            absent.sort_unstable();
            absent.dedup();
            let mut taken = Vec::with_capacity(absent.len());
            for _ in &absent {
                let Some(slot) = self.free_regular(state, None) else {
                    break;
                };
                taken.push(slot);
            }
            absent.truncate(taken.len());
            (absent, taken)
        };
        self.bring(&absent, &taken, true)
    }

    /// The clock's next regular tile of slots that no step named since it
    /// last passed, released: never a landing or pending tile, nor one the
    /// host step `step` names.
    fn free_regular(&self, state: &mut State, step: Option<u32>) -> Option<usize> {
        let (named, pending) = (&state.named, &state.pending);
        let regular = self.shared.slot_tiles;
        state
            .tiles
            .free(|slot| {
                slot >= regular
                    || step == Some(named[slot])
                    || pending.iter().any(|pending| pending.slot == slot)
            })
            .map(|(slot, _)| slot)
    }

    /// The slots of `rows`, named by one step. The tiles of those the device
    /// does not hold are brought to it first, together; they take the place
    /// of tiles no step named since the clock last passed them, never of one
    /// this step names. A tile only landed for a selected read is absent:
    /// a host step is given regular slots only.
    fn slots(&self, rows: &[u64]) -> Result<Vec<i32>, String> {
        let tile = self.tile_rows();
        let (absent, taken) = {
            let mut guard = self.state();
            let state = &mut *guard;
            state.next_step();
            let (mut absent, mut held) = (Vec::new(), 0);
            for &row in rows {
                if !state.tiles.has_row(row) {
                    return Err(format!("table row {row} is outside the table"));
                }
                match state.regular(row - row % tile, self.shared.slot_tiles) {
                    Some(slot) => {
                        state.tiles.use_slot(row);
                        state.served[slot] = true;
                        held += usize::from(
                            std::mem::replace(&mut state.named[slot], state.step) != state.step,
                        );
                    }
                    None => absent.push(row - row % tile),
                }
            }
            absent.sort_unstable();
            absent.dedup();
            if held + absent.len() > self.shared.slot_tiles {
                return Err("a step names more rows than the device has slots for".into());
            }
            let mut taken = Vec::with_capacity(absent.len());
            for _ in &absent {
                let slot = self
                    .free_regular(state, Some(state.step))
                    .ok_or("a streamed table found no slot for a step's row")?;
                state.named[slot] = state.step;
                taken.push(slot);
            }
            (absent, taken)
        };
        self.bring(&absent, &taken, true)?;
        let state = self.state();
        if state.step == Self::KEPT_AFTER_STEPS {
            self.keep(&state);
        }
        rows.iter()
            .map(|row| {
                let slot = state
                    .regular(row - row % tile, self.shared.slot_tiles)
                    .ok_or_else(|| format!("named row {row} is not held"))?;
                Ok((slot as u64 * tile + row % tile) as i32)
            })
            .collect()
    }

    /// Queue the table tiles of first rows `tiles` (ascending) into the
    /// regular slot tiles `taken` (released by the clock, or never given a
    /// tile), in table order into slots in slot order: tiles that follow one another
    /// are read together, and adjacent slots are filled together. While a
    /// read queued before them is not yet filled, the tiles are pending
    /// until the next read queued, the first queued after their conversion;
    /// else every read the filler takes from now on is queued after it, and
    /// they are held at once. `served`: a step named the tiles, now or when
    /// the table was last served.
    fn bring(&self, tiles: &[u64], taken: &[usize], served: bool) -> Result<(), String> {
        if tiles.is_empty() {
            return Ok(());
        }
        let device = |error: seismic::TensorError| format!("device table: {error}");
        let tile = self.tile_rows();
        let mut taken = taken.to_vec();
        taken.sort_unstable();
        let mut runs = Vec::new();
        let mut at = 0;
        while at < taken.len() {
            let run = (at..taken.len())
                .take_while(|next| taken[*next] == taken[at] + (*next - at))
                .count();
            let first = taken[at] as u64 * tile;
            runs.push(TileRun {
                tiles: &tiles[at..at + run],
                slots: self
                    .slots
                    .slice_leading(first, first + run as u64 * tile)
                    .map_err(device)?,
            });
            at += run;
        }
        // The last step's tiles left the staging before these enter it.
        let previous = self.brought.borrow_mut().take();
        if let Some(previous) = previous {
            previous()?;
        }
        // More tiles than a step brings (the load's) are staged in a tensor
        // of their own.
        let rows = tiles.len() as u64 * tile;
        let transient;
        let staging = match rows <= self.staging.extents()[0] {
            true => &self.staging,
            false => {
                transient = staging(&self.slots, self.staging.element(), rows)?;
                &transient
            }
        };
        let mut staged = staging.slice_leading(0, rows).map_err(device)?;
        *self.brought.borrow_mut() = Some(self.source.bring(&runs, &mut staged)?);
        let mut state = self.state();
        let from = state.queued + 1;
        let outstanding = state.filled < state.queued;
        for (first, slot) in tiles.iter().zip(&taken) {
            match outstanding {
                true => state.pending.push(Pending {
                    first: *first,
                    slot: *slot,
                    from,
                    used: served,
                }),
                false => state.tiles.hold(*first, *slot, served),
            }
            state.served[*slot] = served;
        }
        Ok(())
    }

    /// Keep the first row of every regular tile the device holds or brings
    /// that a step named, here or when the table was last served.
    fn keep(&self, state: &State) {
        let Some(kept) = &self.kept else {
            return;
        };
        let mut rows = (0..self.shared.slot_tiles)
            .filter(|slot| state.served[*slot])
            .filter_map(|slot| {
                state.tiles.held(slot).or_else(|| {
                    state
                        .pending
                        .iter()
                        .find(|pending| pending.slot == slot)
                        .map(|pending| pending.first)
                })
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        kept.write(&rows);
    }
}

impl Shared {
    /// The filler: each selected read in queue order.
    fn fill_reads(&self, reads: mpsc::Receiver<(u64, u64)>) {
        for (value, rows) in reads {
            let landed = self.fill(value, rows);
            // The read's outcome is recorded before its gate opens, so its
            // step never completes before it.
            let mut state = self.state.lock().expect("device table lock");
            state.filled = value;
            if let Some(landed) = landed {
                state.landed.insert(value, landed);
            }
            drop(state);
            // A read whose tiles could not be landed opens all the same, so
            // the work behind it drains; its owner fails its step. A gate
            // that cannot open fails the read too.
            if let Err(error) = self.gate.signal(value) {
                self.state
                    .lock()
                    .expect("device table lock")
                    .landed
                    .insert(value, Err(format!("a selected read's gate did not open: {error}")));
            }
        }
    }

    /// Land the tiles read `value` selects (at most `rows`) that the
    /// regular slots do not hold for it: the first rows of the tiles landed,
    /// or why they could not be; `None` of an abandoned read, whose read the
    /// filler signals itself.
    fn fill(&self, value: u64, rows: u64) -> Option<Result<Vec<u64>, String>> {
        // The read before is done with the landing slots and staging, and
        // the device has passed it.
        if value > 1 && !self.read.wait(value - 1, DEVICE_WAIT) {
            return Some(Err("the selected read before never finished".into()));
        }
        // The read's selection reached the host, or the read was abandoned.
        // A read is abandoned before any read queued after it is submitted,
        // so once the notice timeline reached the read's value, by its own
        // notice or a later read's, the abandonment is recorded.
        let started = Instant::now();
        loop {
            let noticed = self.noticed.wait(value, ABANDON_POLL);
            if self
                .state
                .lock()
                .expect("device table lock")
                .abandoned
                .remove(&value)
            {
                // Nobody reads it: its read is done.
                return match self.read.signal(value) {
                    Ok(()) => None,
                    Err(error) => Some(Err(error.to_string())),
                };
            }
            if noticed {
                return Some(self.land(value, rows));
            }
            if started.elapsed() > DEVICE_WAIT {
                return Some(Err("a selected read's selection never reached the host".into()));
            }
        }
    }

    /// Land read `value`'s tiles, once its selection reached the host and
    /// every read before it has read.
    fn land(&self, value: u64, rows: u64) -> Result<Vec<u64>, String> {
        let mut bytes = vec![0u8; rows as usize * 8];
        // SAFETY: the device wrote the selection before it raised `noticed`
        // to `value`; the next read's step writes it only after this read's
        // gate opened.
        unsafe { self.notice.read_shared(0, &mut bytes) }.map_err(|error| error.to_string())?;
        let mut guard = self.state.lock().expect("device table lock");
        let state = &mut *guard;
        // The landing slots hold nothing for this read yet: a tile landed
        // for the read before is absent now. Then the tiles brought for
        // reads from this one on take their slots.
        for landing in self.slot_tiles..self.slot_tiles + self.landing_tiles {
            state.tiles.release(landing);
        }
        let (ready, pending) = std::mem::take(&mut state.pending)
            .into_iter()
            .partition::<Vec<_>, _>(|pending| pending.from <= value);
        state.pending = pending;
        for Pending {
            first, slot, used, ..
        } in ready
        {
            state.tiles.hold(first, slot, used);
        }
        let tile = state.tiles.tile_rows();
        let mut landed = Vec::new();
        for pair in bytes.chunks_exact(8) {
            let token = i32::from_le_bytes(pair[..4].try_into().expect("four bytes"));
            let status = i32::from_le_bytes(pair[4..].try_into().expect("four bytes"));
            // A failed selection names no row: its step's outputs past it
            // are not used, whatever it reads.
            let Ok(row) = u64::try_from(token) else {
                continue;
            };
            if status != 0 || !state.tiles.has_row(row) {
                continue;
            }
            match state.tiles.use_slot(row).map(|slot| (slot / tile) as usize) {
                Some(slot) if slot < self.slot_tiles => state.served[slot] = true,
                // Landed already for this read.
                Some(_) => {}
                None => {
                    let first = row - row % tile;
                    state.tiles.hold(first, self.slot_tiles + landed.len(), true);
                    landed.push(first);
                }
            }
        }
        let tile_bytes = tile as usize * self.row_bytes;
        let mut staged = vec![0u8; landed.len() * tile_bytes];
        for (first, out) in landed.iter().zip(staged.chunks_exact_mut(tile_bytes)) {
            self.rows.read(*first, out)?;
        }
        // SAFETY: the read before has converted and read the landing
        // staging, and this read's conversion waits at its gate.
        unsafe { self.landing_staging.write_shared(0, &staged) }
            .map_err(|error| error.to_string())?;
        // No read is between its gate and its read signal: the directory
        // may be rebuilt, and every change is published at once.
        if state.tiles.crowded() {
            state.tiles.rebuild();
        }
        for (first, bytes) in state.tiles.take_changes() {
            // SAFETY: no read is between its gate and its read signal; the
            // reads before have read, and the ones after wait at their gates.
            unsafe { self.directory.write_shared(first * self.fields * 4, &bytes) }
                .map_err(|error| format!("device table directory: {error}"))?;
        }
        Ok(landed)
    }
}

/// A staging tensor of `rows` rows of the table of `slots`, as `stored`.
fn staging(
    slots: &seismic::Tensor,
    stored: seismic::Element,
    rows: u64,
) -> Result<seismic::Tensor, String> {
    let extents = std::iter::once(rows)
        .chain(slots.extents()[1..].iter().copied())
        .collect::<Vec<_>>();
    // SAFETY: a fill writes every row of its staging before the device
    // reads them.
    unsafe { seismic::Tensor::uninitialized(&slots.device(), stored, &extents) }
        .map_err(|error| format!("device table staging: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, Element};

    /// The three F32 values of table row `row`: `10·row + column`.
    fn row(row: u64) -> Vec<u8> {
        (0..3)
            .flat_map(|column| ((10 * row + column) as f32).to_le_bytes())
            .collect()
    }

    /// Writes each tile's rows into its slots at once; stores rows as they
    /// are held.
    struct Rows;

    impl StoredRows for Rows {
        fn read(&self, first: u64, out: &mut [u8]) -> Result<(), String> {
            for (index, out) in out.chunks_exact_mut(12).enumerate() {
                out.copy_from_slice(&row(first + index as u64));
            }
            Ok(())
        }
    }

    impl TileSource for Rows {
        fn bring(
            &self,
            runs: &[TileRun<'_>],
            staging: &mut seismic::Tensor,
        ) -> Result<TilesBrought, String> {
            let text = |error: seismic::TensorError| error.to_string();
            let bytes = runs
                .iter()
                .flat_map(|run| run.tiles.iter().flat_map(|tile| row(*tile)))
                .collect::<Vec<_>>();
            staging.write_from_host(&bytes).map_err(text)?;
            let mut staged = 0;
            for run in runs {
                let rows = run.tiles.len() as u64;
                let source = staging.slice_leading(staged, staged + rows).map_err(text)?;
                let bytes = source.read_to_host().map_err(|error| error.to_string())?;
                run.slots.clone().write_from_host(&bytes).map_err(text)?;
                staged += rows;
            }
            Ok(Box::new(|| Ok(())))
        }

        fn rows(&self) -> Arc<dyn StoredRows> {
            Arc::new(Rows)
        }
    }

    /// An 8-row, 3-column F32 table on the CPU backend, stored as it is
    /// held, in four slots and two landing slots with staging for a step
    /// that brings one row, seeded with the rows `kept` keeps.
    fn device_table(kept: Option<&Path>) -> Rc<DeviceTable> {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let device = catalog.open_backend(BackendName::Cpu).unwrap();
        let table = DeviceTable::new(
            &device,
            Element::f32(),
            Element::f32(),
            8,
            &[3],
            4,
            2,
            1,
            DirectoryForm::Dense,
            Box::new(Rows),
            kept.map(|directory| KeptRows::of(directory, "table")),
        )
        .unwrap();
        table.seed().unwrap();
        Rc::new(table)
    }

    fn directory(table: &DeviceTable) -> Vec<i32> {
        let mut bytes = vec![0u8; 8 * 4];
        // SAFETY: no device work uses the table.
        unsafe { table.shared.directory.read_shared(0, &mut bytes) }.unwrap();
        bytes
            .chunks_exact(4)
            .map(|slot| i32::from_le_bytes(slot.try_into().unwrap()))
            .collect()
    }

    fn slot(table: &DeviceTable, slot: u64) -> Vec<u8> {
        table
            .slots
            .slice_leading(slot, slot + 1)
            .unwrap()
            .read_to_host()
            .unwrap()
    }

    /// The device's part of selected read `read` of `rows`: its selection
    /// copied for the host, then, once its gate opened, its landing staging
    /// converted into its landing slots and the rows it names read through
    /// the directory, then its read signal.
    fn run(table: &DeviceTable, read: SelectedRead, rows: &[i64]) -> (SubmittedRead, Vec<Vec<u8>>) {
        let selection = rows
            .iter()
            .flat_map(|row| [*row as i32, 0].map(i32::to_le_bytes))
            .flatten()
            .chain(std::iter::repeat(0))
            .take(read.notice.byte_len() as usize)
            .collect::<Vec<_>>();
        // SAFETY: the read's gate has not opened, so the filler does not
        // read the notice yet.
        unsafe { read.notice.write_shared(0, &selection) }.unwrap();
        read.noticed.signal(read.value).unwrap();
        assert!(read.gate.wait(read.value, Duration::from_secs(10)));
        let mut staged = vec![0u8; read.staging.byte_len() as usize];
        // SAFETY: the filler wrote the staging before it opened the gate.
        unsafe { read.staging.read_shared(0, &mut staged) }.unwrap();
        read.landing.clone().write_from_host(&staged).unwrap();
        let found = rows
            .iter()
            .map(|row| {
                let entry = directory(table)[*row as usize];
                assert!(entry >= 0, "row {row} has no slot");
                slot(table, entry as u64)
            })
            .collect();
        read.read.signal(read.value).unwrap();
        (read.submitted(), found)
    }

    #[test]
    fn a_table_held_whole_gathers_by_row() {
        let table = ResidentTable::whole();
        assert!(table.directory().is_none());
        assert_eq!(table.streamed_bytes(), 0);
        assert_eq!(table.indices(&[5, 0, 5]).unwrap(), [5, 0, 5]);
    }

    #[test]
    fn a_streamed_table_brings_rows_to_slots_and_replaces_the_ones_no_step_named_lately() {
        // Before the first step the slots hold the table's first rows,
        // which no step named.
        let streamed = device_table(None);
        assert_eq!(slot(&streamed, 2), row(2));
        let table = ResidentTable::streamed(streamed.clone());
        // Its directory, its one staged row, its two landing staging rows
        // and their notice.
        assert_eq!(table.streamed_bytes(), 8 * 4 + 12 + 2 * 12 + 2 * 8);

        // A step's rows gather by slot. An absent row replaces a row no
        // step named, once however often the step names it: the clock
        // reaches the last of the first rows, the least valuable, first.
        assert_eq!(table.indices(&[6, 1, 6]).unwrap(), [3, 1, 3]);
        assert_eq!(slot(&streamed, 3), row(6));

        // The row a step named (1) is passed over. More rows than the
        // staging holds are brought together.
        assert_eq!(table.indices(&[4, 5]).unwrap(), [0, 2]);
        assert_eq!(slot(&streamed, 0), row(4));
        assert_eq!(slot(&streamed, 2), row(5));

        // Every row was named since the clock last passed: it passes them
        // all once, and replaces the first it meets again that this step
        // does not name.
        assert_eq!(table.indices(&[1, 7]).unwrap(), [1, 3]);

        // A step that names more rows than there are slots, or a row
        // outside the table, changes nothing.
        assert!(table.indices(&[0, 1, 2, 3, 4]).is_err());
        assert!(table.indices(&[8]).is_err());
        assert!(table.indices(&[-1]).is_err());
        assert_eq!(table.indices(&[7, 4, 5, 1]).unwrap(), [3, 0, 2, 1]);
        // The filler publishes the directory: none of this before a read.
        assert_eq!(directory(&streamed), [-1; 8]);
    }

    #[test]
    fn a_selected_read_finds_every_row_it_names() {
        let table = device_table(None);
        // Rows 1 (held) and 5 and 6 (absent): the absent ones are landed.
        let read = table.read_selected(2).unwrap();
        let (submitted, found) = run(&table, read, &[1, 5]);
        assert_eq!(found, [row(1), row(5)]);
        let read = table.read_selected(2).unwrap();
        let (second, found) = run(&table, read, &[6, 6]);
        assert_eq!(found, [row(6), row(6)]);
        // The first read's landed row takes a slot of its own once its step
        // completed; the second's is landed again by the next read.
        submitted.complete().unwrap();
        drop(second);
        let read = table.read_selected(2).unwrap();
        let (submitted, found) = run(&table, read, &[5, 6]);
        assert_eq!(found, [row(5), row(6)]);
        assert_eq!(directory(&table)[5], 3);
        submitted.complete().unwrap();
        // A read beyond the landing slots is refused.
        assert!(table.read_selected(3).is_err());
    }

    #[test]
    fn a_row_brought_for_a_host_step_is_landed_for_reads_queued_before_it() {
        let table = device_table(None);
        // A read is queued; then a host step brings row 6 into slot 3,
        // ahead of work queued after the read. The read must not find it
        // there, nor row 3, which slot 3 held when the read was queued.
        let read = table.read_selected(2).unwrap();
        assert_eq!(ResidentTable::streamed(table.clone()).indices(&[6]).unwrap(), [3]);
        let (submitted, found) = run(&table, read, &[6, 3]);
        assert_eq!(found, [row(6), row(3)]);
        assert!(directory(&table)[6] >= 4, "row 6 was landed");
        drop(submitted);
        // The next read finds row 6 in slot 3.
        let read = table.read_selected(1).unwrap();
        let (_, found) = run(&table, read, &[6]);
        assert_eq!(found, [row(6)]);
        assert_eq!(directory(&table)[6], 3);
    }

    #[test]
    fn an_abandoned_read_lets_the_reads_after_it_through() {
        let table = device_table(None);
        let abandoned = table.read_selected(1).unwrap();
        let read = table.read_selected(1).unwrap();
        drop(abandoned);
        let (_, found) = run(&table, read, &[7]);
        assert_eq!(found, [row(7)]);
    }

    #[test]
    fn the_rows_steps_named_are_held_first_the_next_time() {
        let kept = std::env::temp_dir().join(format!(
            "magnitude-device-table-{}",
            std::process::id()
        ));
        let file = KeptRows::of(&kept, "table").0;
        let table = ResidentTable::streamed(device_table(Some(&kept)));
        table.indices(&[6, 1]).unwrap();
        // The rows are kept when the table is dropped, and once after its
        // first steps.
        assert!(!file.exists());
        for _ in 1..DeviceTable::KEPT_AFTER_STEPS {
            table.indices(&[1]).unwrap();
        }
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "1 6");
        table.indices(&[3]).unwrap();
        drop(table);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "1 3 6");

        // The kept rows take the first slots.
        let table = ResidentTable::streamed(device_table(Some(&kept)));
        assert_eq!(table.indices(&[1, 3, 6, 0]).unwrap(), [0, 1, 2, 3]);
        drop(table);
        std::fs::remove_dir_all(kept).unwrap();
    }
}
