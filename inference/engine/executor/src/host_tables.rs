//! Host-resident gathered tables (model-family plan §3.7).
//!
//! A table too large to be a device weight and read only by row gather (the
//! Gemma per-layer embedding: 1.3–1.6 GB, 5–6 KB per token) stays in its
//! artifact file and is read through the host page cache. Its bytes are a
//! claim in the system-RAM domain: the pages a served model reads stay
//! resident. Each step gathers the source bytes of every batch row's table
//! row (row m = the table row of token m) into the step's upload, in the
//! table's source representation; the graph converts them into the resident
//! row layout on the device.

use crate::residency::Stored;
use magnitude_artifacts::FileSource;
use seismic::Element;
use std::fmt;
use std::sync::Arc;

/// One table: `rows` rows of `row_bytes` source bytes each from `offset`.
#[derive(Clone)]
pub struct HostTable {
    file: Arc<FileSource>,
    offset: u64,
    rows: u64,
    columns: u64,
    row_bytes: usize,
    source: Element,
}

#[derive(Debug)]
pub enum HostTableError {
    /// The table is not a matrix, or its source bytes are not whole rows.
    Shape(String),
    Read(magnitude_artifacts::Error),
    /// A batch row names a table row outside the table.
    Row { row: i64, rows: u64 },
    /// The gather buffer does not hold the batch's rows exactly.
    Buffer { expected: usize, actual: usize },
}

impl fmt::Display for HostTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(formatter, "host table shape: {message}"),
            Self::Read(error) => write!(formatter, "host table read: {error}"),
            Self::Row { row, rows } => {
                write!(formatter, "host table row {row} is outside its {rows} rows")
            }
            Self::Buffer { expected, actual } => write!(
                formatter,
                "host table gather buffer holds {actual} bytes where {expected} are gathered"
            ),
        }
    }
}

impl std::error::Error for HostTableError {}

impl HostTable {
    /// Map the stored `[rows, columns]` table.
    pub fn open(stored: &Stored) -> Result<Self, HostTableError> {
        let &[rows, columns] = stored.shape() else {
            return Err(HostTableError::Shape(format!(
                "a gathered table is a matrix, not {:?}",
                stored.shape()
            )));
        };
        let source = stored
            .source_element()
            .ok_or_else(|| HostTableError::Shape("the table's encoding has no element".into()))?;
        let row_bytes = source
            .canonical_byte_len(&[1, columns])
            .map_err(|error| HostTableError::Shape(error.to_string()))?;
        let (file, offset, length) = stored.file_range();
        if rows.checked_mul(row_bytes) != Some(length) {
            return Err(HostTableError::Shape(format!(
                "{length} source bytes are not {rows} rows of {row_bytes}"
            )));
        }
        let row_bytes = usize::try_from(row_bytes)
            .map_err(|_| HostTableError::Shape("a table row exceeds the host range".into()))?;
        Ok(Self {
            file: Arc::clone(file),
            offset,
            rows,
            columns,
            row_bytes,
            source,
        })
    }

    /// The bytes the table claims in the system-RAM domain.
    pub fn bytes(&self) -> u64 {
        self.rows * self.row_bytes as u64
    }

    /// The source representation the gathered rows are uploaded in.
    pub fn source(&self) -> Element {
        self.source
    }

    pub fn columns(&self) -> u64 {
        self.columns
    }

    /// Source bytes of one gathered row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Copy the source bytes of `rows` (table row indices, one per batch
    /// row, in batch order) into `out`, which holds exactly that many rows.
    pub fn gather(&self, rows: &[i64], out: &mut [u8]) -> Result<(), HostTableError> {
        let expected = rows.len() * self.row_bytes;
        if out.len() != expected {
            return Err(HostTableError::Buffer {
                expected,
                actual: out.len(),
            });
        }
        for (&row, destination) in rows.iter().zip(out.chunks_exact_mut(self.row_bytes)) {
            let index = u64::try_from(row)
                .ok()
                .filter(|index| *index < self.rows)
                .ok_or(HostTableError::Row {
                    row,
                    rows: self.rows,
                })?;
            self.file
                .read_into(self.offset + index * self.row_bytes as u64, destination)
                .map_err(HostTableError::Read)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::StoredTensor;
    use seismic::DType;

    /// A 5-row, 3-column F32 table after a 16-byte prefix, whose value at
    /// (row, column) is `10·row + column`.
    fn table(name: &str) -> (std::path::PathBuf, HostTable) {
        let path = std::env::temp_dir().join(format!(
            "magnitude-host-table-{name}-{}.bin",
            std::process::id()
        ));
        let mut bytes = vec![0xee; 16];
        for row in 0..5u32 {
            for column in 0..3u32 {
                bytes.extend(((10 * row + column) as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let stored = Stored::Dense(StoredTensor {
            source: Arc::new(FileSource::open(&path).unwrap()),
            offset: 16,
            nbytes: 5 * 3 * 4,
            dtype: DType::F32,
            shape: vec![5, 3],
        });
        (path, HostTable::open(&stored).unwrap())
    }

    fn values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn gathers_each_batch_rows_table_row_in_batch_order() {
        let (path, table) = table("order");
        assert_eq!(table.bytes(), 60);
        assert_eq!(table.row_bytes(), 12);
        let mut out = vec![0; 3 * 12];
        table.gather(&[4, 0, 4], &mut out).unwrap();
        assert_eq!(
            values(&out),
            [40.0, 41.0, 42.0, 0.0, 1.0, 2.0, 40.0, 41.0, 42.0]
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_rows_outside_the_table_and_misfit_buffers() {
        let (path, table) = table("bounds");
        let mut out = vec![0; 12];
        assert!(matches!(
            table.gather(&[5], &mut out),
            Err(HostTableError::Row { row: 5, rows: 5 })
        ));
        assert!(matches!(
            table.gather(&[-1], &mut out),
            Err(HostTableError::Row { row: -1, .. })
        ));
        assert!(matches!(
            table.gather(&[0, 1], &mut out),
            Err(HostTableError::Buffer { expected: 24, actual: 12 })
        ));
        std::fs::remove_file(path).unwrap();
    }
}
