//! CUDA storage exposed through the generic device service: device memory,
//! pinned host memory mapped into the device address space, or the backed
//! prefix of a reserved address range of either.

use crate::blocks::Range;
use crate::driver::{self, DriverError, HostMapped, ReservedMemory};
use seismic_reservation::Reserved;
use std::sync::Arc;

#[derive(Clone)]
pub struct Buffer {
    memory: Arc<Memory>,
}

enum Memory {
    /// A range of one of the device's blocks.
    Device(Range),
    /// Host-written inputs: filled by the host without a driver call.
    Mapped(HostMapped),
    /// The committed bytes of a reservation, which are backed.
    /// Its `Resizer` changes them in place.
    Reserved(Arc<Reserved<ReservedMemory>>),
}

impl Buffer {
    pub(crate) fn device(range: Range) -> Self {
        Self {
            memory: Arc::new(Memory::Device(range)),
        }
    }
    pub(crate) fn mapped(mapped: HostMapped) -> Self {
        Self {
            memory: Arc::new(Memory::Mapped(mapped)),
        }
    }
    pub(crate) fn reserved(reservation: Arc<Reserved<ReservedMemory>>) -> Self {
        Self {
            memory: Arc::new(Memory::Reserved(reservation)),
        }
    }
    /// The reservation this buffer is a backed prefix of.
    pub(crate) fn reservation(&self) -> Option<&ReservedMemory> {
        match &*self.memory {
            Memory::Reserved(reservation) => Some(reservation.backing()),
            Memory::Device(_) | Memory::Mapped(_) => None,
        }
    }
    pub fn len(&self) -> u64 {
        match &*self.memory {
            Memory::Device(range) => range.bytes(),
            Memory::Mapped(mapped) => mapped.bytes as u64,
            Memory::Reserved(reservation) => reservation.state().bytes,
        }
    }
    /// The host address of host memory's first byte (mapped memory, or a
    /// reservation of host memory, which the host maps at the device's
    /// addresses); device memory has none (the host reaches it by driver
    /// copies).
    pub(crate) fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        match &*self.memory {
            Memory::Mapped(mapped) => std::ptr::NonNull::new(mapped.host()),
            Memory::Reserved(reservation) => {
                let memory = reservation.backing();
                memory.memory.host_addressed().then(|| std::ptr::NonNull::new(memory.base as *mut u8)).flatten()
            }
            Memory::Device(_) => None,
        }
    }
    /// The bytes no driver copy of this buffer's crosses, from its start
    /// ([`ReservedMemory::copy_span`]); none where a copy may span it whole.
    fn copy_span(&self) -> Option<u64> {
        self.reservation().and_then(|reservation| reservation.copy_span())
    }
    /// The address kernels use.
    pub fn pointer(&self) -> u64 {
        match &*self.memory {
            Memory::Device(range) => range.pointer(),
            Memory::Mapped(mapped) => mapped.device,
            Memory::Reserved(reservation) => reservation.backing().base,
        }
    }
    /// Synchronous host write. Device memory is written by a driver copy
    /// on the non-blocking transfer stream, ordered only by the caller's
    /// access fence; mapped memory is written in place.
    pub(crate) fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<(), DriverError> {
        match &*self.memory {
            Memory::Device(range) => range.upload_at(offset, bytes),
            Memory::Mapped(mapped) => {
                mapped.write_at(offset, bytes);
                Ok(())
            }
            Memory::Reserved(reservation) => {
                assert_within(offset, bytes.len(), reservation.state().bytes as usize, "upload_at");
                match self.host_address() {
                    // SAFETY: within the backed bytes, which the host maps.
                    Some(base) => unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.as_ptr().add(offset), bytes.len());
                        Ok(())
                    },
                    None => driver::upload(
                        reservation.backing().context(),
                        reservation.backing().base + offset as u64,
                        bytes,
                    ),
                }
            }
        }
    }
    /// Synchronous host read.
    pub(crate) fn download_at(&self, offset: usize, bytes: &mut [u8]) -> Result<(), DriverError> {
        match &*self.memory {
            Memory::Device(range) => range.download_at(offset, bytes),
            Memory::Mapped(mapped) => {
                mapped.read_at(offset, bytes);
                Ok(())
            }
            Memory::Reserved(reservation) => {
                assert_within(offset, bytes.len(), reservation.state().bytes as usize, "download_at");
                match self.host_address() {
                    // SAFETY: within the backed bytes, which the host maps.
                    Some(base) => unsafe {
                        std::ptr::copy_nonoverlapping(base.as_ptr().add(offset), bytes.as_mut_ptr(), bytes.len());
                        Ok(())
                    },
                    None => driver::download(
                        reservation.backing().context(),
                        reservation.backing().base + offset as u64,
                        bytes,
                    ),
                }
            }
        }
    }
}

/// This crate's own precondition: host access stays within the backed bytes.
fn assert_within(offset: usize, length: usize, backed: usize, operation: &str) {
    assert!(
        offset.checked_add(length).is_some_and(|end| end <= backed),
        "Buffer::{operation} precondition: [{offset}, {offset}+{length}) exceeds {backed} backed bytes"
    );
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}

/// The driver copies of `bytes` bytes from `source` at `source_offset` to
/// `destination` at `destination_offset`, in order, as (source offset,
/// destination offset, bytes): one, cut wherever either side's bytes leave
/// one of its copy spans ([`Buffer::copy_span`]).
pub(crate) fn copy_pieces(
    source: &Buffer,
    source_offset: u64,
    destination: &Buffer,
    destination_offset: u64,
    bytes: u64,
) -> Vec<(u64, u64, u64)> {
    cut(&[(source_offset, source.copy_span()), (destination_offset, destination.copy_span())], bytes)
        .into_iter()
        .map(|(done, piece)| (source_offset + done, destination_offset + done, piece))
        .collect()
}

/// The driver fills of `bytes` bytes of `destination` at `offset`, in
/// order, as (offset, bytes): one, cut wherever they leave one of its copy
/// spans, which bound a fill as they bound a copy.
pub(crate) fn fill_pieces(destination: &Buffer, offset: u64, bytes: u64) -> Vec<(u64, u64)> {
    cut(&[(offset, destination.copy_span())], bytes)
        .into_iter()
        .map(|(done, piece)| (offset + done, piece))
        .collect()
}

/// `bytes` bytes run from each of `sides`' offsets, cut wherever a side's
/// bytes leave one of its spans (a span from its buffer's start, or none):
/// the pieces as (bytes done before it, its bytes), in order.
fn cut(sides: &[(u64, Option<u64>)], bytes: u64) -> Vec<(u64, u64)> {
    let mut pieces = Vec::new();
    let mut done = 0;
    while done < bytes {
        let room = sides
            .iter()
            .map(|&(offset, span)| span.map_or(u64::MAX, |span| span - (offset + done) % span))
            .min()
            .unwrap_or(u64::MAX);
        let piece = (bytes - done).min(room);
        pieces.push((done, piece));
        done += piece;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::cut;

    /// Pieces end exactly at each side's span boundaries, from any offset
    /// within a span, and cover the bytes once, in order.
    #[test]
    fn copies_are_cut_at_every_side_s_span() {
        const SPAN: u64 = 2 << 20;
        // Unbounded sides: one piece, whatever the offsets.
        assert_eq!(cut(&[(123, None), (SPAN - 1, None)], 5 * SPAN), vec![(0, 5 * SPAN)]);
        assert_eq!(cut(&[(0, Some(SPAN))], 0), vec![]);
        // Within one span: one piece.
        assert_eq!(cut(&[(SPAN + 16, Some(SPAN))], SPAN - 16), vec![(0, SPAN - 16)]);
        // Mid-span start over two boundaries.
        let at = SPAN / 2 + 3;
        assert_eq!(
            cut(&[(at, Some(SPAN))], 2 * SPAN),
            vec![(0, SPAN - at), (SPAN - at, SPAN), (2 * SPAN - at, at)]
        );
        // Two bounded sides at different phases: cut at both.
        let pieces = cut(&[(SPAN - 100, Some(SPAN)), (SPAN - 300, Some(SPAN))], SPAN);
        assert_eq!(pieces, vec![(0, 100), (100, 200), (300, SPAN - 300)]);
        for (sides, bytes) in [(vec![(7u64, Some(4096u64)), (4095, Some(8192))], 100_000u64), (vec![(1, Some(3))], 10)] {
            let pieces = cut(&sides, bytes);
            assert_eq!(pieces.iter().map(|piece| piece.1).sum::<u64>(), bytes);
            let mut done = 0;
            for &(start, piece) in &pieces {
                assert_eq!(start, done, "pieces in order with no gap");
                for &(offset, span) in &sides {
                    let span = span.unwrap();
                    assert_eq!((offset + start) / span, (offset + start + piece - 1) / span, "no piece crosses a span");
                }
                done += piece;
            }
        }
    }
}
