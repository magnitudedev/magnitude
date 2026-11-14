//! Streamed host tables.
//!
//! A host table (`host_tables`) claims its whole byte range in system RAM:
//! every row a served model may read stays in the OS file cache. A streamed
//! table claims a row cache instead. Its steps run the graphs a host
//! table's steps run, and gather the same source bytes.
//!
//! The row cache keeps the source rows of the latest steps, one to a slot.
//! A step's rows reach its upload one of two ways, by which costs it less:
//!
//! - a step most of whose rows the cache holds assembles them in a buffer
//!   the steps share: a held row is copied from its slot, and any other is
//!   read as a host table reads it and copied into a slot;
//! - a step most of whose rows are new takes as many adjacent slots and
//!   gathers into them: a new row is read straight into its slot as a host
//!   table reads it, a held one is moved there, and the step uploads the
//!   slots themselves.
//!
//! Either way a row the cache holds is not read again, and a step of new
//! rows alone copies nothing a host table's step does not.
//!
//! The slots and the shared buffer are host pages of their own, made once
//! at import, charged as they are made to the ledger entry the load claims
//! its host tables under, and freed from it as they return to the OS: the
//! one record of their bytes.
//!
//! A table a device kernel gathers is streamed on the device instead
//! (`device_table`).
//!
//! Nothing here knows which table it streams or what its consumer computes.

use crate::host_tables::{HostTable, HostTableError};
use crate::slot_directory::SlotDirectory;
use seismic::memory::{Backed, Entry, HostBlock, PageSize, Wiring};
use std::sync::{Arc, Mutex};

/// The rows a streamed table holds: a host table's row cache in host memory,
/// a device table's slots on the device. Many times the distinct rows a
/// conversation names, and at least the rows one step can name on every
/// backend; a table of no more rows is held whole.
pub const STREAMED_TABLE_ROWS: u64 = 16_384;

/// The rows the row cache of a streamed host table of `rows` rows holds.
pub fn cached_rows(rows: u64) -> u64 {
    rows.min(STREAMED_TABLE_ROWS)
}

/// The row cache's bytes: `len` bytes at the start of a host block charged,
/// whole pages, to the entry it was taken from.
struct CacheBytes {
    block: Backed<HostBlock>,
    len: usize,
}

impl CacheBytes {
    /// `len` zero bytes in a block of their own charged to `entry`, its
    /// whole pages, in one change.
    fn take(len: usize, entry: &Arc<dyn Entry>) -> Result<Self, HostTableError> {
        let block = Backed::take(entry, len as u64, Wiring::Pageable, |charging| {
            let block = HostBlock::allocate(len as u64, PageSize::Host).map_err(HostTableError::Unallocated)?;
            charging.took_more(block.bytes() as u64 - len as u64);
            Ok::<_, HostTableError>(block)
        })?;
        Ok(Self { block, len })
    }
}

impl std::ops::Deref for CacheBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the block holds at least `len` bytes, backed and
        // initialized, and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.block.base().as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for CacheBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `deref`; the block is this value's alone, so the
        // exclusive borrow of it is the only access.
        unsafe { std::slice::from_raw_parts_mut(self.block.base().as_ptr(), self.len) }
    }
}

/// The source rows of the latest steps, one to a slot.
struct RowCache {
    row_bytes: usize,
    slots: SlotDirectory,
    /// The slots' rows, then the buffer steps assemble their rows in, one
    /// after another, of the most rows a step gathers.
    bytes: CacheBytes,
    /// The bytes of the slots, where the buffer begins.
    slot_bytes: usize,
}

impl RowCache {
    fn new(slots: usize, step_rows: usize, table: &HostTable, entry: &Arc<dyn Entry>) -> Result<Self, HostTableError> {
        let slot_bytes = slots * table.row_bytes();
        Ok(Self {
            row_bytes: table.row_bytes(),
            slots: SlotDirectory::new(
                table.rows(),
                slots as u64,
                1,
                crate::planning::DirectoryForm::Dense,
            ),
            bytes: CacheBytes::take(slot_bytes + step_rows * table.row_bytes(), entry)?,
            slot_bytes,
        })
    }

    /// The source bytes of `rows`, one per batch row, in batch order.
    fn gather(&mut self, table: &HostTable, rows: &[i64]) -> Result<&[u8], HostTableError> {
        let mut held = 0;
        for row in rows {
            held += usize::from(self.slots.slot(table.row_index(*row)?).is_some());
        }
        // Moving a held row costs more than assembling it, and assembling
        // a new row more than reading it into place.
        match (2 * held < rows.len())
            .then(|| self.slots.free_adjacent(rows.len()))
            .flatten()
        {
            Some(first) => self.place(table, rows, first),
            None => self.assemble(table, rows),
        }
    }

    /// Gather `rows` into the free adjacent slots from `first`.
    fn place(
        &mut self,
        table: &HostTable,
        rows: &[i64],
        first: usize,
    ) -> Result<&[u8], HostTableError> {
        let row_bytes = self.row_bytes;
        for (slot, row) in (first..).zip(rows) {
            let index = table.row_index(*row)?;
            let destination = slot * row_bytes;
            match self.slots.use_slot(index) {
                Some(source) => {
                    let source = source as usize;
                    self.bytes
                        .copy_within(source * row_bytes..(source + 1) * row_bytes, destination);
                    self.slots.release(source);
                }
                None => table.gather(
                    std::slice::from_ref(row),
                    &mut self.bytes[destination..destination + row_bytes],
                )?,
            }
            self.slots.hold(index, slot, true);
        }
        Ok(&self.bytes[first * row_bytes..(first + rows.len()) * row_bytes])
    }

    /// Gather `rows` into the shared buffer.
    ///
    /// # Panics
    /// Where a step names more rows than the buffer holds: more than its
    /// launch's row class.
    fn assemble(&mut self, table: &HostTable, rows: &[i64]) -> Result<&[u8], HostTableError> {
        let row_bytes = self.row_bytes;
        let (held, buffer) = self.bytes.split_at_mut(self.slot_bytes);
        let holds = buffer.len() / row_bytes;
        let assembled = buffer.get_mut(..rows.len() * row_bytes).unwrap_or_else(|| {
            panic!("invariant violated: a step assembles {} rows in a buffer of {holds}", rows.len())
        });
        for (row, destination) in rows.iter().zip(assembled.chunks_exact_mut(row_bytes)) {
            let index = table.row_index(*row)?;
            if let Some(slot) = self.slots.use_slot(index) {
                let slot = slot as usize;
                destination.copy_from_slice(&held[slot * row_bytes..(slot + 1) * row_bytes]);
                continue;
            }
            table.gather(std::slice::from_ref(row), destination)?;
            if let Some((slot, _)) = self.slots.free(|_| false) {
                held[slot * row_bytes..(slot + 1) * row_bytes].copy_from_slice(destination);
                self.slots.hold(index, slot, true);
            }
        }
        Ok(&*assembled)
    }
}

pub struct StreamedTable {
    table: HostTable,
    rows: Mutex<RowCache>,
}

impl StreamedTable {
    /// `table` streamed through a row cache of [`cached_rows`] rows and a
    /// buffer of `step_rows`, the most rows a step gathers, charged to
    /// `entry`: the load's host-tables claim's.
    pub fn new(table: HostTable, step_rows: usize, entry: &Arc<dyn Entry>) -> Result<Self, HostTableError> {
        let cached = cached_rows(table.rows()) as usize;
        Self::caching(table, cached, step_rows, entry)
    }

    /// `new`, with a row cache of `cached_rows` rows.
    fn caching(table: HostTable, cached_rows: usize, step_rows: usize, entry: &Arc<dyn Entry>) -> Result<Self, HostTableError> {
        Ok(Self {
            rows: Mutex::new(RowCache::new(cached_rows, step_rows, &table, entry)?),
            table,
        })
    }

    /// The table's rows.
    pub fn rows(&self) -> u64 {
        self.table.rows()
    }

    /// Hand `take` the source bytes of `rows` (table row indices, one per
    /// batch row, in batch order).
    pub fn gather<T>(
        &self,
        rows: &[i64],
        take: impl FnOnce(&[u8]) -> T,
    ) -> Result<T, HostTableError> {
        let mut cache = self.rows.lock().expect("row cache lock");
        cache.gather(&self.table, rows).map(take)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::{Stored, StoredTensor};
    use magnitude_artifacts::FileSource;
    use seismic::DType;
    use std::sync::Arc;

    const ROWS: u32 = 8;

    /// The three F32 values of table row `row`: `10·row + column`.
    fn row(row: u32) -> Vec<u8> {
        (0..3)
            .flat_map(|column| ((10 * row + column) as f32).to_le_bytes())
            .collect()
    }

    /// An 8-row, 3-column F32 table after a 16-byte prefix.
    fn table(name: &str) -> (std::path::PathBuf, HostTable) {
        let path = std::env::temp_dir().join(format!(
            "magnitude-streamed-table-{name}-{}.bin",
            std::process::id()
        ));
        let mut bytes = vec![0xee; 16];
        bytes.extend((0..ROWS).flat_map(row));
        std::fs::write(&path, &bytes).unwrap();
        let stored = Stored::dense(StoredTensor {
            source: Arc::new(FileSource::open(&path).unwrap()),
            offset: 16,
            nbytes: u64::from(ROWS) * 12,
            dtype: DType::F32,
            shape: vec![u64::from(ROWS), 3],
        });
        (path, HostTable::open(&stored).unwrap())
    }

    #[test]
    fn a_streamed_table_gathers_a_host_tables_bytes_with_any_cache() {
        let (path, table) = table("gather");
        let entry: Arc<dyn Entry> = Arc::new(seismic::memory::Tally::unbounded(seismic::StoreIdentity::HostRam));
        let steps: [&[i64]; 9] = [
            &[4, 0, 4],
            &[1],
            &[1],
            &[7, 6, 5, 4, 3],
            &[0, 1],
            &[1, 0, 1, 0],
            &[2],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[7, 7, 7],
        ];
        for cached_rows in [0, 1, 2, 3, 5, 8] {
            let streamed = StreamedTable::caching(table.clone(), cached_rows, ROWS as usize, &entry).unwrap();
            for rows in steps {
                let mut expected = vec![0; rows.len() * table.row_bytes()];
                table.gather(rows, &mut expected).unwrap();
                let gathered = streamed.gather(rows, <[u8]>::to_vec).unwrap();
                assert_eq!(gathered, expected, "{cached_rows} cached rows, step {rows:?}");
            }
            assert!(streamed.gather(&[8], <[u8]>::to_vec).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }
}
