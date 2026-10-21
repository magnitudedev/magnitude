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
use std::sync::Arc;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
};

/// A gathered table as its readers find it on the device.
///
/// A reader whose rows the host knows gathers by `indices`. A reader whose
/// rows the device selects finds their slots in the directory and says so
/// when it is queued (`read_selected`); what it is given tells, once the host
/// knows the rows, whether the device held them, and brings the ones it did
/// not hold (`SelectedRead::observe`).
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
    /// table's directory and staging, a stored table's directory.
    pub(crate) fn visit_extra_storage(&self, visit: &mut dyn FnMut(&seismic::Tensor)) {
        match self {
            Self::Whole => {}
            Self::Streamed(table) => {
                visit(&table.directory);
                visit(&table.staging);
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

    /// A streamed table's directory (`gathered_row`).
    pub fn directory(&self) -> Option<&seismic::Tensor> {
        match self {
            Self::Whole => None,
            Self::Streamed(table) => Some(&table.directory),
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

    /// What a streamed table allocates beside its slots: its directory and
    /// its staging (`WeightPlan::streamed_bytes`).
    pub fn streamed_bytes(&self) -> u64 {
        match self {
            Self::Whole => 0,
            Self::Streamed(table) => {
                table.directory.storage_bytes() + table.staging.storage_bytes()
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

    /// A reader of rows the device selects is being queued.
    pub fn read_selected(&self) -> SelectedRead {
        SelectedRead {
            queued: match self {
                Self::Streamed(table) => Some((table.clone(), table.queue_reader())),
                Self::Whole | Self::Stored(_) => None,
            },
        }
    }
}

/// A queued read of rows the device selected.
pub struct SelectedRead {
    /// The streamed table read and the count of its changes then.
    queued: Option<(Rc<DeviceTable>, u64)>,
}

impl SelectedRead {
    /// Whether the read found table row `row`: always of a table held whole;
    /// of a streamed one, when the device has held the row since the read
    /// was queued.
    pub fn found(&self, row: u64) -> bool {
        self.queued
            .as_ref()
            .is_none_or(|(table, changes)| table.held_since(row, *changes))
    }

    /// The rows the read selected, once the host knows them: a streamed
    /// table serves the ones it holds and brings the others, so that later
    /// reads find them (`DeviceTable::observe`).
    pub fn observe(&self, rows: &[u64]) -> Result<(), String> {
        match &self.queued {
            Some((table, _)) => table.observe(rows),
            None => Ok(()),
        }
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
}

/// Waits for queued tiles to reach their slots, after which their staging
/// holds nothing the device reads.
pub type TilesBrought = Box<dyn FnOnce() -> Result<(), String>>;

/// A streamed gathered table: the device holds slots for only some of the
/// table's row tiles (a tile is the rows its storage lays out together; one
/// row for most storage) and a directory of every row's slot.
///
/// The directory changes only while no reader of it is queued: a change that
/// follows one waits for the device to pass it. A reader therefore sees the
/// slots and the directory as they were when it was queued.
pub struct DeviceTable {
    /// The slots: the leading rows of the table's shape.
    slots: seismic::Tensor,
    directory: seismic::Tensor,
    /// The rows a step's tiles pass through, as stored.
    staging: seismic::Tensor,
    source: Box<dyn TileSource>,
    kept: Option<KeptRows>,
    state: RefCell<State>,
}

struct State {
    tiles: SlotDirectory,
    /// Per tile of slots, whether a step ever named its tile, here or when
    /// the table was last served: the tiles kept for the next time.
    served: Vec<bool>,
    /// Per tile of slots, the last step that named its tile.
    named: Vec<u32>,
    /// Per tile of slots, `changes` when its tile was brought.
    brought_at: Vec<u64>,
    /// The steps queued so far, with known or selected rows.
    step: u32,
    /// The tiles of slots given a tile before the first step, the first
    /// ones.
    filled: usize,
    /// The times tiles were brought.
    changes: u64,
    /// Whether a reader of the directory was queued since the device last
    /// passed everything queued.
    reader_queued: bool,
    /// The tiles the last step brought, while the device may still read
    /// their staging.
    brought: Option<TilesBrought>,
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

impl Drop for DeviceTable {
    fn drop(&mut self) {
        self.keep(&self.state.borrow());
    }
}

impl DeviceTable {
    /// The steps after which a table's rows are kept: a process that ends
    /// without dropping the table still leaves the rows its first steps
    /// named.
    const KEPT_AFTER_STEPS: u32 = 256;

    /// A table of `rows` rows, each of `element` and `row_shape` and stored
    /// as `stored`, in `slots` slots on `device`, with staging for steps
    /// that bring up to `staging_rows` rows. Its slots are empty (`seed`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &seismic::Device,
        element: seismic::Element,
        stored: seismic::Element,
        rows: u64,
        row_shape: &[u64],
        slots: u64,
        staging_rows: u64,
        form: DirectoryForm,
        source: Box<dyn TileSource>,
        kept: Option<KeptRows>,
    ) -> Result<Self, String> {
        let tile = element.tile_rows();
        if rows % tile != 0 || slots % tile != 0 || staging_rows % tile != 0 || slots == 0 {
            return Err(format!(
                "a table of {rows} rows in {slots} slots staged {staging_rows} rows at a time is not whole tiles of {tile} rows"
            ));
        }
        let shape = |rows: u64| {
            std::iter::once(rows)
                .chain(row_shape.iter().copied())
                .collect::<Vec<_>>()
        };
        let mut tiles = SlotDirectory::new(rows, slots, tile, form);
        let slot_tiles = tiles.slot_tiles();
        let directory_shape = [form.entries(rows, slots), form.fields()];
        let slots = seismic::Tensor::zeros(device, element, &shape(slots))
            .map_err(|error| format!("device table slots: {error}"))?;
        Ok(Self {
            staging: staging(&slots, stored, staging_rows)?,
            slots,
            directory: seismic::Tensor::from_host(
                device,
                seismic::Element::i32(),
                &directory_shape,
                &tiles.entry_bytes(),
            )
            .map_err(|error| format!("device table directory: {error}"))?,
            source,
            kept,
            state: RefCell::new(State {
                tiles,
                served: vec![false; slot_tiles],
                named: vec![0; slot_tiles],
                brought_at: vec![0; slot_tiles],
                step: 0,
                filled: 0,
                changes: 0,
                reader_queued: false,
                brought: None,
            }),
        })
    }

    pub fn slots_tensor(&self) -> &seismic::Tensor {
        &self.slots
    }

    /// Fill the slots before the table's first step, the rows worth most
    /// first: the rows steps named when the table was last served, which
    /// the clock passes once before it replaces them, then the table's
    /// first rows.
    pub fn seed(&self) -> Result<(), String> {
        let kept = self.kept.as_ref().map_or_else(Vec::new, KeptRows::read);
        self.hold(kept, true)?;
        self.hold(0..self.slots.extents()[0], false)
    }

    /// Bring the tiles of `rows` the device does not hold into the slots no
    /// tile was given yet, as many as those take, and wait for them: the
    /// table's first step finds the device idle. Rows outside the table name
    /// none. `served`: steps named these rows when the table was last
    /// served.
    fn hold(&self, rows: impl IntoIterator<Item = u64>, served: bool) -> Result<(), String> {
        let mut state = self.state.borrow_mut();
        let tile = state.tiles.tile_rows();
        let mut absent = rows
            .into_iter()
            .filter(|row| state.tiles.has_row(*row) && state.tiles.slot(*row).is_none())
            .map(|row| row - row % tile)
            .collect::<Vec<_>>();
        absent.sort_unstable();
        absent.dedup();
        // Once a step named rows the slots are the steps'.
        let free = match state.step {
            0 => state.tiles.slot_tiles() - state.filled,
            _ => 0,
        };
        absent.truncate(free);
        let taken = (state.filled..state.filled + absent.len()).collect::<Vec<_>>();
        state.filled += absent.len();
        self.bring(&mut state, &absent, &taken, served)?;
        match state.brought.take() {
            Some(brought) => brought(),
            None => Ok(()),
        }
    }

    /// A reader that gathers through the directory is being queued: the
    /// count of the table's changes it reads after.
    fn queue_reader(&self) -> u64 {
        let mut state = self.state.borrow_mut();
        state.next_step();
        state.reader_queued = true;
        if state.step == Self::KEPT_AFTER_STEPS {
            // This read's rows are still device-selected. Keep the rows
            // observed from the preceding steps without waiting for it.
            self.keep(&state);
        }
        state.changes
    }

    /// Whether a reader queued after `changes` changes found table row
    /// `row`: the device has held it since. A step named the row.
    fn held_since(&self, row: u64, changes: u64) -> bool {
        let mut state = self.state.borrow_mut();
        let Some(slot) = state.tiles.use_slot(row) else {
            return false;
        };
        let slot = (slot / state.tiles.tile_rows()) as usize;
        state.served[slot] = true;
        state.brought_at[slot] <= changes
    }

    /// The rows a reader of the directory selected, once the host knows
    /// them. The ones held are served, as a step's are; the tiles of the
    /// others are brought, together, into tiles of slots no step named since
    /// the clock last passed them, as a step's absent rows are (`slots`).
    /// The read missed them; later reads find them. A selection whose tiles
    /// outnumber the slots the clock frees brings as many as it frees: never
    /// one the current step names, nor one this selection holds.
    fn observe(&self, rows: &[u64]) -> Result<(), String> {
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        let tile = state.tiles.tile_rows();
        let mut absent = Vec::new();
        for &row in rows {
            if !state.tiles.has_row(row) {
                return Err(format!("table row {row} is outside the table"));
            }
            match state.tiles.use_slot(row) {
                Some(slot) => {
                    let slot = (slot / tile) as usize;
                    state.served[slot] = true;
                    state.named[slot] = state.step;
                }
                None => absent.push(row - row % tile),
            }
        }
        absent.sort_unstable();
        absent.dedup();
        let mut taken = Vec::with_capacity(absent.len());
        for _ in &absent {
            let (named, step) = (&state.named, state.step);
            let Some((slot, _)) = state.tiles.free(|slot| named[slot] == step) else {
                break;
            };
            state.named[slot] = state.step;
            taken.push(slot);
        }
        absent.truncate(taken.len());
        taken.sort_unstable();
        self.bring(state, &absent, &taken, true)
    }

    /// The slots of `rows`, named by one step. The tiles of those the device
    /// does not hold are brought to it first, together; they take the place
    /// of tiles no step named since the clock last passed them, never of one
    /// this step names.
    fn slots(&self, rows: &[u64]) -> Result<Vec<i32>, String> {
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        state.next_step();
        let tile = state.tiles.tile_rows();
        let (mut absent, mut held) = (Vec::new(), 0);
        for &row in rows {
            if !state.tiles.has_row(row) {
                return Err(format!("table row {row} is outside the table"));
            }
            match state.tiles.use_slot(row) {
                Some(slot) => {
                    let slot = (slot / tile) as usize;
                    state.served[slot] = true;
                    held += usize::from(std::mem::replace(&mut state.named[slot], state.step) != state.step);
                }
                None => absent.push(row - row % tile),
            }
        }
        absent.sort_unstable();
        absent.dedup();
        if held + absent.len() > state.tiles.slot_tiles() {
            return Err("a step names more rows than the device has slots for".into());
        }
        let mut taken = Vec::with_capacity(absent.len());
        for _ in &absent {
            let (named, step) = (&state.named, state.step);
            let (slot, _) = state
                .tiles
                .free(|slot| named[slot] == step)
                .ok_or("a streamed table found no slot for a step's row")?;
            state.named[slot] = state.step;
            taken.push(slot);
        }
        taken.sort_unstable();
        self.bring(state, &absent, &taken, true)?;
        if state.step == Self::KEPT_AFTER_STEPS {
            self.keep(state);
        }
        Ok(rows
            .iter()
            .map(|row| state.tiles.slot(*row).expect("a named row is held") as i32)
            .collect())
    }

    /// Queue the table tiles of first rows `tiles` (ascending) into the slot
    /// tiles `taken` (ascending), which the directory released. Tiles in
    /// table order into slots in slot order: tiles that follow one another
    /// are read together, and adjacent slots are filled together. `served`:
    /// a step named the tiles, now or when the table was last served.
    fn bring(
        &self,
        state: &mut State,
        tiles: &[u64],
        taken: &[usize],
        served: bool,
    ) -> Result<(), String> {
        if tiles.is_empty() {
            return Ok(());
        }
        let device = |error: seismic::TensorError| format!("device table: {error}");
        let tile = state.tiles.tile_rows();
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
        // The last step's tiles left the staging before these enter it.
        if let Some(brought) = state.brought.take() {
            brought()?;
        }
        let brought = self.source.bring(&runs, &mut staged)?;
        // Behind a queued reader the directory waits until the device has
        // passed it and filled the slots.
        match std::mem::take(&mut state.reader_queued) {
            true => brought()?,
            false => state.brought = Some(brought),
        }
        // The directory's entries that changed (the tiles that left their
        // slots and the ones brought) are written: a dense directory's once,
        // from the first row that changed to the last.
        state.changes += 1;
        for (first, slot) in tiles.iter().zip(taken) {
            state.tiles.hold(*first, *slot, served);
            state.served[*slot] = served;
            state.brought_at[*slot] = state.changes;
        }
        for (first, bytes) in state.tiles.take_changes() {
            let entries = (bytes.len() / 4) as u64 / self.directory.extents()[1];
            self.directory
                .slice_leading(first, first + entries)
                .map_err(device)?
                .write_from_host(&bytes)
                .map_err(device)?;
        }
        Ok(())
    }

    /// Keep the first row of every tile the device holds that a step named,
    /// here or when the table was last served.
    fn keep(&self, state: &State) {
        let Some(kept) = &self.kept else {
            return;
        };
        let mut rows = (0..state.tiles.slot_tiles())
            .filter(|slot| state.served[*slot])
            .filter_map(|slot| state.tiles.held(slot))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        kept.write(&rows);
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

    /// Writes each tile's rows into its slots at once.
    struct Rows;

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
    }

    /// An 8-row, 3-column F32 table on the CPU backend, stored as it is
    /// held, in four slots with staging for a step that brings one row,
    /// seeded with the rows `kept` keeps.
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
        table
            .directory
            .read_to_host()
            .unwrap()
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

    #[test]
    fn a_table_held_whole_gathers_by_row_and_finds_every_selected_row() {
        let table = ResidentTable::whole();
        assert!(table.directory().is_none());
        assert_eq!(table.streamed_bytes(), 0);
        assert_eq!(table.indices(&[5, 0, 5]).unwrap(), [5, 0, 5]);
        assert!(table.read_selected().found(7));
    }

    #[test]
    fn a_streamed_table_brings_rows_to_slots_and_replaces_the_ones_no_step_named_lately() {
        // Before the first step the slots hold the table's first rows,
        // which no step named.
        let table = device_table(None);
        assert_eq!(directory(&table), [0, 1, 2, 3, -1, -1, -1, -1]);
        assert_eq!(slot(&table, 2), row(2));
        let table = ResidentTable::streamed(table);
        let ResidentTable::Streamed(streamed) = &table else {
            unreachable!("a streamed table");
        };
        // Its directory and its one staged row.
        assert_eq!(table.streamed_bytes(), 8 * 4 + 12);

        // A step's rows gather by slot. An absent row replaces a row no
        // step named, once however often the step names it: the clock
        // reaches the last of the first rows, the least valuable, first.
        assert_eq!(table.indices(&[6, 1, 6]).unwrap(), [3, 1, 3]);
        assert_eq!(directory(streamed), [0, 1, 2, -1, -1, -1, 3, -1]);
        assert_eq!(slot(streamed, 3), row(6));

        // The row a step named (1) is passed over. More rows than the
        // staging holds are brought together.
        assert_eq!(table.indices(&[4, 5]).unwrap(), [0, 2]);
        assert_eq!(directory(streamed), [-1, 1, -1, -1, 0, 2, 3, -1]);
        assert_eq!(slot(streamed, 0), row(4));
        assert_eq!(slot(streamed, 2), row(5));

        // Every row was named since the clock last passed: it passes them
        // all once, and replaces the first it meets again that this step
        // does not name.
        assert_eq!(table.indices(&[1, 7]).unwrap(), [1, 3]);
        assert_eq!(directory(streamed), [-1, 1, -1, -1, 0, 2, -1, 3]);

        // A step that names more rows than there are slots, or a row
        // outside the table, changes nothing.
        assert!(table.indices(&[0, 1, 2, 3, 4]).is_err());
        assert!(table.indices(&[8]).is_err());
        assert!(table.indices(&[-1]).is_err());
        assert_eq!(directory(streamed), [-1, 1, -1, -1, 0, 2, -1, 3]);
        assert_eq!(table.indices(&[7, 4, 5, 1]).unwrap(), [3, 0, 2, 1]);
    }

    #[test]
    fn a_selected_read_finds_the_rows_the_device_held_when_it_was_queued() {
        let table = ResidentTable::streamed(device_table(None));
        let read = table.read_selected();
        assert!(read.found(1));
        assert!(!read.found(5));
        // The step that brings the row puts it on the device for reads
        // queued from now, not for the one queued before.
        table.indices(&[5]).unwrap();
        assert!(!read.found(5));
        assert!(table.read_selected().found(5));
        assert!(!read.found(8));
    }

    #[test]
    fn a_selected_read_brings_the_rows_it_missed_for_the_reads_after_it() {
        let table = ResidentTable::streamed(device_table(None));
        let ResidentTable::Streamed(streamed) = &table else {
            unreachable!("a streamed table");
        };
        let read = table.read_selected();
        assert!(!read.found(5));
        // The read selected rows 1 (held) and 5 and 6 (missed): the missed
        // rows take tiles no step named, never the one the read held.
        read.observe(&[1, 5, 6, 5]).unwrap();
        assert_eq!(directory(streamed), [0, 1, -1, -1, -1, 2, 3, -1]);
        assert_eq!(slot(streamed, 2), row(5));
        assert!(!read.found(5));
        let next = table.read_selected();
        assert!(next.found(1) && next.found(5) && next.found(6));
        // Every tile is replaced once none is the current step's.
        next.observe(&[0, 2, 4, 7]).unwrap();
        let held = table.read_selected();
        assert!([0, 2, 4, 7].iter().all(|row| held.found(*row)));
        // A selection beyond the slots the clock frees brings what it can,
        // keeping the row it held.
        held.observe(&[0, 1, 3, 5, 6]).unwrap();
        let after = table.read_selected();
        assert!([0, 1, 3, 5].iter().all(|row| after.found(*row)));
        assert!(!after.found(6));
        assert!(after.observe(&[8]).is_err());
        // A table held whole holds every row.
        ResidentTable::whole().read_selected().observe(&[8]).unwrap();
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
        assert!(table.read_selected().found(2));
        // The rows are kept when the table is dropped, and once after its
        // first steps.
        assert!(!file.exists());
        for _ in 1..DeviceTable::KEPT_AFTER_STEPS {
            table.indices(&[1]).unwrap();
        }
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "1 2 6");
        table.indices(&[3]).unwrap();
        drop(table);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "1 2 3 6");

        // The kept rows take the first slots.
        let table = device_table(Some(&kept));
        assert_eq!(directory(&table), [-1, 0, 1, 2, -1, -1, 3, -1]);
        drop(table);
        std::fs::remove_dir_all(kept).unwrap();
    }

    #[test]
    fn selected_reads_keep_observed_rows_before_the_table_is_dropped() {
        let kept = std::env::temp_dir().join(format!(
            "magnitude-device-table-selected-{}",
            std::process::id()
        ));
        let file = KeptRows::of(&kept, "table").0;
        let table = ResidentTable::streamed(device_table(Some(&kept)));
        table.indices(&[6]).unwrap();
        for _ in 1..DeviceTable::KEPT_AFTER_STEPS - 1 {
            assert!(table.read_selected().found(2));
        }
        assert!(!file.exists());
        assert!(table.read_selected().found(2));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "2 6");
        drop(table);
        std::fs::remove_dir_all(kept).unwrap();
    }
}
