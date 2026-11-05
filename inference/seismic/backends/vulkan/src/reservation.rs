//! Reserved storage: a reservation committed chunk by chunk, each chunk its
//! own memory object and buffer with a device address, which device work
//! reaches only through the reservation's record table (entry `r` the
//! device address of record `r` in its chunk; no record crosses one). The
//! resize between commitments is the one every granule-backed reservation
//! shares (`seismic_reservation`); this is its Vulkan backing
//! ([`Chunks`]).
//!
//! Chunks are backed as `Facts::chunk_backing` says. Device memory
//! chunks (of a device whose memory is its own) are zeroed
//! on the device's second queue ([`Facts::reservation`]): a growth is
//! prepared beside work the first queue holds parked for the host, so
//! nothing it does may queue behind that work. The others lie in one
//! reserved host address range, so the host addresses the reservation in
//! place at one base: host pages imported (`VK_EXT_external_memory_host`),
//! whose fresh pages read zero; or, on a device whose memory is the host's,
//! device memory of its kind mapped at its place in the range
//! (`VK_EXT_map_memory_placed`), which the host zeroes there. The table is upload
//! storage, host-mapped, so a chunk's entries are written in place as it is
//! formed, beside device work that reads the others.

use crate::device::{Device, Inner};
use crate::facts::FILL_UNIT;
use crate::facts::ChunkBacking;
use crate::memory::{Buffer, Chunk, Mapping, Pool};
use crate::Facts;
use seismic_compiler::errors::ExecutionError;
use seismic_reservation::{Backing, HostRange, PageSize, Pages, Reserved, Resizer};
use seismic_lang::registry::{chunk_entries, record_table_bytes, record_table_entry};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// The Vulkan backing of a reservation: its chunks, in order, one per
/// backed granule, and its record table.
pub struct Chunks {
    device: Arc<Inner>,
    /// What its chunks are charged to: the accounts of the open that
    /// reserved it.
    accounts: Arc<crate::memory::MemoryAccounts>,
    granule: usize,
    granules: usize,
    /// Bytes of one record, a divisor of the granule.
    record: u64,
    // Dropped before `host`: a chunk is freed (a placed one unmapped back to
    // the range) before its pages are unmapped.
    chunks: Mutex<Vec<Chunk>>,
    /// What backs its chunks, and the host range they lie in where the host
    /// addresses them.
    backing: Backed,
    /// One device address a record of the reservation, written as chunks
    /// are formed: entries of released chunks are never read.
    table: Arc<Buffer>,
}

pub type VulkanResizer = Resizer<Chunks>;

/// What backs a reservation's chunks ([`ChunkBacking`]), with the reserved
/// host range they lie in where the host addresses them.
enum Backed {
    Device(Arc<Pool>),
    HostPages(Pages),
    Placed(Pages, Arc<Pool>),
}

impl Backed {
    /// The host range its chunks lie in; none for device memory chunks.
    fn range(&self) -> Option<&Pages> {
        match self {
            Self::Device(_) => None,
            Self::HostPages(pages) | Self::Placed(pages, _) => Some(pages),
        }
    }
}

impl Chunks {
    fn chunks(&self) -> std::sync::MutexGuard<'_, Vec<Chunk>> {
        self.chunks.lock().expect("reservation chunks lock is never poisoned")
    }

    /// Chunk `index`: device memory of its kind; its granule's host pages
    /// imported; or device memory of its kind mapped at its granule's place
    /// in the host range.
    fn form(&self, index: usize) -> Result<Chunk, ExecutionError> {
        let memory = &self.device.memory;
        let bytes = self.granule as u64;
        let at = |pages: &Pages| NonNull::new(pages.0.at(index)).expect("a reserved range is never at null");
        let block = match &self.backing {
            Backed::Device(pool) => memory.chunk(&self.device.device, pool, bytes, Mapping::Pool, &self.accounts)?,
            Backed::HostPages(pages) => {
                let pointer = at(pages);
                let importable = self.device.importable(pointer)?;
                memory.import(&self.device.device, pointer, bytes, importable, &self.accounts)?
            }
            Backed::Placed(pages, pool) => {
                let pointer = at(pages);
                let map = |memory| self.device.map_placed(memory, pointer);
                memory.chunk(&self.device.device, pool, bytes, Mapping::Placed(pointer, &map), &self.accounts)?
            }
        };
        Ok(Chunk::new(self.device.clone(), block, matches!(self.backing, Backed::Placed(..))))
    }

    /// Write the table entries of the records of `chunks`, the chunks of
    /// granules from `first`.
    fn enter(&self, first: usize, chunks: &[Chunk]) {
        let records = self.table.bytes() / seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES;
        for (index, chunk) in (first as u64..).zip(chunks) {
            let (start, entries) =
                chunk_entries(index, self.granule as u64, self.record, records, chunk.range().address());
            self.table.range().write_mapped(record_table_entry(start), &entries);
        }
    }
}

fn host_refused(failure: seismic_reservation::AllocationFailure) -> ExecutionError {
    ExecutionError::AllocationFailed(format!("a Vulkan reservation's host pages: {failure}"))
}

impl Backing for Chunks {
    type Error = ExecutionError;
    type View = ();
    type Change = ();
    type Retired = ();

    fn granule(&self) -> usize {
        self.granule
    }

    fn granules(&self) -> usize {
        self.granules
    }

    /// Form chunks `[from, to)`, zero, with their records' table entries.
    /// On failure nothing new stays formed.
    fn back(&self, from: usize, to: usize) -> Result<(), ExecutionError> {
        if from >= to {
            return Ok(());
        }
        let imported = match &self.backing {
            Backed::HostPages(pages) => Some(pages),
            Backed::Device(_) | Backed::Placed(..) => None,
        };
        if let Some(pages) = imported {
            pages.back(from, to).map_err(host_refused)?;
        }
        let formed = seismic_reservation::time_backing(seismic_reservation::BackingPhase::Pin, || {
            (from..to).map(|index| self.form(index)).collect::<Result<Vec<_>, _>>()
        });
        // Fresh host pages read zero; device memory chunks are zeroed whole
        // on the device, placed ones by the host through the range.
        let granule = self.granule as u64;
        let zeroed = formed.and_then(|formed| match &self.backing {
            Backed::HostPages(_) => Ok(formed),
            Backed::Placed(pages, _) => {
                seismic_reservation::time_backing(seismic_reservation::BackingPhase::Zero, || pages.0.zero(from as u64 * granule..to as u64 * granule));
                Ok(formed)
            }
            Backed::Device(_) => {
                let regions = formed.iter().map(|chunk| (chunk.range(), 0, granule)).collect::<Vec<_>>();
                seismic_reservation::time_backing(seismic_reservation::BackingPhase::Zero, || self.device.zero(&regions)).map(|()| formed)
            }
        });
        let formed = match zeroed {
            Ok(formed) => formed,
            Err(error) => {
                if let Some(pages) = imported {
                    pages.unback(from, to);
                }
                return Err(error);
            }
        };
        seismic_reservation::time_backing(seismic_reservation::BackingPhase::Pin, || self.enter(from, &formed));
        let mut chunks = self.chunks();
        assert_eq!(chunks.len(), from, "a reservation backs granules in order");
        chunks.extend(formed);
        Ok(())
    }

    /// Free chunks `[from, to)`, the tail, then the host pages they
    /// imported.
    fn unback(&self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let mut chunks = self.chunks();
        assert_eq!(chunks.len(), to, "a reservation releases its tail granules");
        chunks.truncate(from);
        drop(chunks);
        // A placed chunk unmapped back to the range as it dropped; imported
        // pages are released.
        if let Backed::HostPages(pages) = &self.backing {
            pages.unback(from, to);
        }
    }

    /// Host pages are written in place; device chunks are filled on the
    /// second queue, each overlapped one over its share. `bytes` lie on
    /// record boundaries, so they are whole words (records are,
    /// [`Device::allocate_reserved`]).
    fn zero(&self, bytes: Range<u64>) -> Result<(), ExecutionError> {
        if let Some(pages) = self.backing.range() {
            pages.0.zero(bytes);
            return Ok(());
        }
        let granule = self.granule as u64;
        let chunks = self.chunks();
        let regions = (bytes.start / granule..bytes.end.div_ceil(granule))
            .map(|index| {
                let start = bytes.start.max(index * granule);
                let end = bytes.end.min((index + 1) * granule);
                (chunks[index as usize].range(), start - index * granule, end - start)
            })
            .collect::<Vec<_>>();
        self.device.zero(&regions)
    }

    fn view(&self, _: usize) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn change(&self, _: usize, _: usize) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn apply(&self, _: &mut (), _: ()) {}
}

/// The committed prefix of a Vulkan reservation, resized in place
/// ([`VulkanResizer::prepare`]): chunks device work reaches only through
/// the record table ([`Self::table`]), never bound.
#[derive(Clone)]
pub struct ReservedBuffer {
    reservation: Arc<Reserved<Chunks>>,
}

impl ReservedBuffer {
    /// Committed bytes.
    pub fn len(&self) -> u64 {
        self.reservation.state().bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn chunks(&self) -> &Chunks {
        self.reservation.backing()
    }

    /// The record table device work reaches the records through, which the
    /// reservation formed with it and keeps as its chunks change.
    pub fn table(&self) -> Arc<Buffer> {
        self.chunks().table.clone()
    }

    /// The device address of byte `offset`, in the chunk holding it, which
    /// must be backed; its address is fixed while it is.
    pub fn address(&self, offset: u64) -> u64 {
        let granule = self.chunks().granule as u64;
        self.chunks().chunks()[(offset / granule) as usize].range().address() + offset % granule
    }

    /// The host address of byte 0 where the host addresses the reservation
    /// in place (host storage's pages, whose range never moves).
    pub fn host_base(&self) -> Option<NonNull<u8>> {
        self.chunks().backing.range().map(|pages| pages.0.base())
    }

    /// Each chunk's share of committed bytes `[offset, offset + len)`:
    /// `(chunk, offset in it, offset in the span, bytes)`.
    pub(crate) fn spans(&self, offset: u64, len: usize, mut each: impl FnMut(&Chunk, u64, usize, usize) -> Result<(), ExecutionError>) -> Result<(), ExecutionError> {
        let end = offset.checked_add(len as u64).filter(|&end| end <= self.len());
        assert!(end.is_some(), "Vulkan reserved access [{offset}, +{len}) exceeds the committed {} bytes", self.len());
        let granule = self.chunks().granule as u64;
        let chunks = self.chunks().chunks();
        let mut at = offset;
        while at < offset + len as u64 {
            let index = at / granule;
            let within = at % granule;
            let bytes = (granule - within).min(offset + len as u64 - at);
            each(&chunks[index as usize], within, (at - offset) as usize, bytes as usize)?;
            at += bytes;
        }
        Ok(())
    }

    /// Write `bytes` at `offset` of the committed bytes: through the host
    /// pages, or chunk by chunk as the device writes host data
    /// ([`Device::write`]).
    pub fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        let device = Device {
            inner: self.chunks().device.clone(),
            accounts: self.chunks().accounts.clone(),
        };
        self.spans(offset, bytes.len(), |chunk, within, at, len| {
            device.write_range(chunk.range(), within, &bytes[at..at + len])
        })
    }

    /// Read committed bytes from `offset` into `into`.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        let device = Device {
            inner: self.chunks().device.clone(),
            accounts: self.chunks().accounts.clone(),
        };
        let len = into.len();
        self.spans(offset, len, |chunk, within, at, bytes| {
            device.read_range(chunk.range(), within, &mut into[at..at + bytes])
        })
    }
}

impl Device {
    /// Zero-filled storage over a reservation of `reserved` bytes in `kinds`,
    /// in order, of `record`-byte records in `granule`-byte chunks (whole
    /// records and whole chunk units, none smaller than the device's
    /// granule, [`Facts::reservation`]), the leading `committed` bytes
    /// backed, and its resizer, which changes the backed prefix in place.
    /// Device memory chunks' records are whole words, which their zero
    /// fills write. Each chunk is an allocation: `maxMemoryAllocationSize`
    /// bounds a chunk (the runtime admits it), not the reservation.
    pub fn allocate_reserved(
        &self,
        kinds: &[u32],
        committed: u64,
        reserved: u64,
        granule: u64,
        record: u64,
    ) -> Result<(ReservedBuffer, VulkanResizer), ExecutionError> {
        let facts: &Facts = self.facts();
        let &[first, ..] = kinds else {
            return Err(ExecutionError::AllocationFailed("a reservation names no kind".into()));
        };
        let form = facts
            .reservation(first)
            .map_err(|reason| ExecutionError::AllocationFailed(format!("this Vulkan device reserves nothing in kind {first}: {reason}")))?
            .ok_or_else(|| ExecutionError::AllocationFailed(format!("this Vulkan device reserves nothing in kind {first}")))?;
        let words = form.host_addressed || record % FILL_UNIT == 0;
        if granule < form.granule || granule % form.chunk_unit != 0 || record == 0 || granule % record != 0 || !words {
            return Err(ExecutionError::AllocationFailed(format!(
                "a {granule}-byte chunk of {record}-byte records is not whole {}-byte units and whole \
                 records (of whole words in device memory chunks), at least {} bytes",
                form.chunk_unit, form.granule
            )));
        }
        let unrepresentable = || ExecutionError::AllocationFailed(format!("a {reserved}-byte reservation overflows"));
        let granules = usize::try_from(reserved.max(committed).div_ceil(granule).max(1)).map_err(|_| unrepresentable())?;
        let table_bytes = record_table_bytes(reserved / record).ok_or_else(unrepresentable)?;
        let table = Arc::new(self.allocate_upload(table_bytes, seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES, None)?);
        let range = || HostRange::reserve(reserved.max(committed), granule, PageSize::Large).map(Pages).map_err(host_refused);
        let backing = match facts.chunk_backing(first).expect("a kind the device reserves in has a backing") {
            ChunkBacking::Device => Backed::Device(self.inner.memory.pool(kinds)?),
            ChunkBacking::HostPages => Backed::HostPages(range()?),
            ChunkBacking::Placed => Backed::Placed(range()?, self.inner.memory.pool(kinds)?),
        };
        let chunks = Chunks {
            device: self.inner.clone(),
            accounts: self.accounts.clone(),
            backing,
            granule: usize::try_from(granule).map_err(|_| unrepresentable())?,
            granules,
            record,
            chunks: Mutex::new(Vec::new()),
            table,
        };
        let (reservation, resizer) = Reserved::new(chunks, committed)?;
        Ok((ReservedBuffer { reservation }, resizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES;
    use seismic_topology::{place, Placing, ReservedContract};

    /// The table entries `[first, first + count)` of `buffer`, as written.
    fn entries(buffer: &ReservedBuffer, first: u64, count: u64) -> Vec<u64> {
        let mut bytes = vec![0; (count * RECORD_TABLE_ENTRY_BYTES) as usize];
        buffer.table().range().read_mapped(record_table_entry(first), &mut bytes);
        bytes
            .chunks_exact(RECORD_TABLE_ENTRY_BYTES as usize)
            .map(|entry| u64::from_le_bytes(entry.try_into().unwrap()))
            .collect()
    }

    /// On every device, in each kind placement reserves a contract in (a
    /// report's kind may reserve in form while its device backs no storage
    /// buffer there, which only an open device says): a reservation commits
    /// zeroed chunks, grows chunk by chunk keeping what it holds, its new
    /// bytes zero and each new chunk's records entered in the table at
    /// their addresses; a shrink keeps the leading chunk, and regrowing
    /// into it reads zero past the shrink.
    #[test]
    fn a_reservation_grows_chunk_by_chunk_through_its_table() {
        let mut exercised = 0;
        for description in crate::test_devices() {
            let device = Device::open(description.facts.uuid, crate::test_accounts(&description.facts)).expect("a device meeting the floor opens");
            let facts = device.facts();
            let kinds = crate::topology::kinds(facts);
            let placed = ReservedContract::ALL
                .map(|contract| place(&kinds, facts.device_type.memory_is_host(), Placing::Reserved(contract)));
            for storage in placed.iter().filter_map(|kinds| kinds.first().copied()) {
                let form = facts.reservation(storage).expect("reservation capability query").expect("a kind that reserves");
                exercised += 1;
                let granule = form.granule;
                let (record, per) = (granule / 4, 4);
                let (buffer, resizer) = device
                    .allocate_reserved(&[storage], granule, 4 * granule, granule, record)
                    .expect("a reservation");
                assert_eq!(buffer.host_base().is_some(), form.host_addressed);
                let read = |offset: u64, bytes: u64| {
                    let mut into = vec![0xa5; bytes as usize];
                    buffer.read(offset, &mut into).expect("a host read");
                    into
                };
                assert!(read(0, granule).iter().all(|&byte| byte == 0), "committed chunks read zero");
                let pattern = (0..granule).map(|index| (index % 251) as u8 + 1).collect::<Vec<_>>();
                buffer.write(0, &pattern).expect("a host write");
                let addresses = |first: u64, count: u64| {
                    (first..first + count).map(|entry| buffer.address(entry * record)).collect::<Vec<_>>()
                };
                assert_eq!(entries(&buffer, 0, per), addresses(0, per));

                let resizer = resizer.prepare(3 * granule).map_err(|(_, error)| error).unwrap().install().release();
                assert_eq!(buffer.len(), 3 * granule);
                assert_eq!(read(0, granule), pattern, "a growth keeps what the reservation holds");
                assert!(read(granule, 2 * granule).iter().all(|&byte| byte == 0), "grown chunks read zero");
                assert_eq!(entries(&buffer, 0, 3 * per), addresses(0, 3 * per));

                let resizer = resizer.prepare(record).map_err(|(_, error)| error).unwrap().install().release();
                assert_eq!(read(0, record), pattern[..record as usize]);
                let _resizer = resizer.prepare(granule).map_err(|(_, error)| error).unwrap().install().release();
                assert_eq!(read(0, record), pattern[..record as usize], "a shrink keeps its leading records");
                assert!(read(record, granule - record).iter().all(|&byte| byte == 0), "regrown bytes read zero");
            }
        }
    }
}
