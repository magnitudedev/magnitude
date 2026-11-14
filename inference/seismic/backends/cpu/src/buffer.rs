//! Host storage for one global allocation.
//!
//! A `Buffer` this crate allocates is a range of one [`PageBlock`]: host
//! pages of its own, never the heap's. A buffer over
//! [`CARVED_LIMIT`] has its block alone; any other is carved from the one
//! block its ledger entry is filling ([`Carver`]), as fresh pages of its own
//! cost a page fault and a zeroing per allocation, tens to hundreds of
//! times a heap allocation of the same size, which is why the carver holds
//! the block it fills across buffers. A block of its own returns to the OS at once as its last
//! buffer drops, so a free reads at once as the process's memory falling,
//! never as others' rising behind an allocator that keeps them; a carved
//! block's return with its entry; and a block's charge is its whole pages,
//! exactly what the OS holds for it.
//!
//! Kernels receive a buffer's address through the launch frame; the runtime
//! reads and writes it through `DeviceService`. The runtime's submission
//! discipline is single-threaded per device: no host access overlaps a
//! launch that binds the buffer, which is what makes the raw address a
//! valid kernel operand.
//!
//! Every byte a buffer takes from the host is charged to the ledger entry
//! its allocation names, and freed from it as the buffer's last handle
//! drops.

use seismic_compiler::errors::ExecutionError;
use seismic_memory::entry::{Backed, Entry, Wiring};
use seismic_memory::{AllocationFailure, HostBlock, PageSize};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

/// The pages a [`Carver`] block holds at least: what an entry's small
/// buffers share before it fills another.
const CARVED_BLOCK_PAGES: u64 = 16;

/// The largest buffer carved from an entry's held block; a larger one takes
/// pages of its own. Faulting fresh pages costs far more than zeroing held
/// ones, so every buffer a heap serves from memory it keeps is carved, and
/// the heap keeps what is under its mmap threshold ceiling (32 MiB); this
/// is 1 MiB because the block an entry holds idle is as large as the
/// largest buffer carved, and a buffer of more (weights, state) lives long
/// enough for its faults to amortize.
const CARVED_LIMIT: u64 = 1 << 20;

/// The least alignment of every buffer's first byte, whatever its request:
/// the widest SIMD register the backend emits, as the heap's `max_align_t`
/// and every device's allocation base are a floor no request lowers, so
/// host code that reads a buffer's base in words wider than its elements
/// (an i32 tensor's u64 pairs, a vector of lanes) reads it aligned, and no
/// two buffers share one such word.
pub(crate) const LEAST_ALIGNMENT: u64 = crate::profile::SimdTier::Bits512.bits() as u64 / 8;

/// A host allocation the OS refused, or one no host layout represents.
pub(crate) fn allocation_error(error: AllocationFailure) -> ExecutionError {
    ExecutionError::AllocationFailed(error.to_string())
}

enum Allocation {
    /// A range of host pages of this crate's ([`WritableBuffer::new`]).
    Pages(PageRange),
    /// Host memory another owner holds ([`Buffer::borrowed`]).
    Borrowed(BorrowedRegion),
}

/// Host pages of their own, every one backed and zero, charged whole to the
/// entry they name and returned to the OS when the last range over them
/// drops: the pages unmap, then their charge is freed, in one change
/// ([`Backed`]).
struct PageBlock(Backed<HostBlock>);

impl PageBlock {
    /// `bytes` (whole host pages) charged to `entry` and allocated, in one
    /// change.
    fn allocate(bytes: u64, entry: &Arc<dyn Entry>) -> Result<Arc<Self>, ExecutionError> {
        let block = Backed::take(entry, bytes, Wiring::Pageable, |_| {
            let block = HostBlock::allocate(bytes, PageSize::Host).map_err(allocation_error)?;
            assert_eq!(block.bytes() as u64, bytes, "invariant violated: a block of whole host pages is that many bytes");
            Ok::<_, ExecutionError>(block)
        })?;
        Ok(Arc::new(Self(block)))
    }

    fn base(&self) -> NonNull<u8> {
        self.0.base()
    }

    fn bytes(&self) -> u64 {
        self.0.bytes() as u64
    }

    fn entry(&self) -> &Arc<dyn Entry> {
        self.0.entry()
    }
}

/// `bytes` logical bytes at `offset` in a block, which the range keeps
/// alive.
struct PageRange {
    /// Taken as the range drops ([`Carver::returned`]).
    block: Option<Arc<PageBlock>>,
    offset: u64,
    /// The logical bytes exposed to the runtime and transfer methods.
    bytes: u64,
}

/// The one owner of how a small buffer takes pages: the block each entry
/// is filling, in which each buffer lies after the one before it. The
/// carver holds the block, so a buffer made and dropped with no sibling
/// alive reuses its pages instead of mapping, faulting and unmapping a
/// block of its own (the cost pooling exists to remove). A full block is
/// replaced and returns as its buffers drop; an idle one is rewound and
/// zeroed in place for the next buffer. The block's charge is the entry's
/// whole pages while the carver holds it; it returns as the entry's holder
/// is gone ([`Entry::on_close`]) with no buffer in the block, or as the
/// last buffer drops once it is, inside that free, never left until an
/// unrelated carve. An entry with no holder (a tally) is closed from the
/// start, so its block returns with its last buffer and nothing idles for
/// the process's life.
#[derive(Default)]
struct Carver {
    filling: Mutex<HashMap<usize, Filling>>,
}

struct Filling {
    block: Arc<PageBlock>,
    /// The bytes of the block already handed out.
    used: u64,
}

impl Drop for PageRange {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            Carver::process().returned(block);
        }
    }
}

impl PageRange {
    fn block(&self) -> &Arc<PageBlock> {
        self.block.as_ref().expect("a range holds its block until it drops")
    }
}

impl Filling {
    /// No buffer lies in the block: the carver holds the only handle, and
    /// handles are only cloned under its lock.
    fn idle(&self) -> bool {
        Arc::strong_count(&self.block) == 1
    }
}

impl Carver {
    fn process() -> &'static Self {
        static CARVER: OnceLock<Carver> = OnceLock::new();
        CARVER.get_or_init(Self::default)
    }

    /// An entry's identity: its block's charge holds it, so a held block's
    /// key is never another entry's.
    fn key(entry: &Arc<dyn Entry>) -> usize {
        Arc::as_ptr(entry).cast::<()>() as usize
    }

    /// `entry`'s holder is gone: its idle block returns now, the free it
    /// makes inside the entry's change. A block with buffers returns as
    /// the last of them drops ([`PageRange`]).
    fn closed(&self, key: usize) {
        let idle = {
            let mut filling = self.filling.lock().expect("the carver's lock is never held across a panic");
            filling.get(&key).is_some_and(Filling::idle).then(|| filling.remove(&key))
        };
        drop(idle);
    }

    /// A range's handle on its block dropped, under the lock, so the
    /// count of handles is whole: where the entry's holder is gone
    /// already and the carver's idle block is all that is left of it, the
    /// block returns with this handle, inside this free; the entry's
    /// closing returns it otherwise ([`Self::closed`]).
    fn returned(&self, block: Arc<PageBlock>) {
        let key = Self::key(block.entry());
        let mut filling = self.filling.lock().expect("the carver's lock is never held across a panic");
        if block.entry().closed() && filling.get(&key).is_some_and(|held| Arc::ptr_eq(&held.block, &block) && Arc::strong_count(&block) == 2) {
            filling.remove(&key);
        }
        drop(block);
        drop(filling);
    }

    /// `bytes` (at most [`CARVED_LIMIT`], at least one byte's room) at
    /// `alignment` from `entry`'s filling block, a new one (of the larger of
    /// the request's whole pages and [`CARVED_BLOCK_PAGES`]) when it has no
    /// room. Where buffers keep the filling block, it stays the entry's
    /// filling block unless the new one has more room left, so a block a
    /// buffer fills alone never retires the room of one whose buffers keep
    /// it (the weights' 64 KiB blocks held 1.28 MB idle behind 128 and 160
    /// KiB buffers that each took a block of their own); an idle one, which
    /// returns whole as it is replaced, is replaced, so a buffer made and
    /// dropped with nothing beside it reuses its block.
    fn carve(&self, bytes: u64, alignment: u64, page: u64, entry: &Arc<dyn Entry>) -> Result<PageRange, ExecutionError> {
        let key = Self::key(entry);
        let mut filling = self.filling.lock().expect("the carver's lock is never held across a panic");
        if let Some(current) = filling.get_mut(&key) {
            if current.idle() && current.used != 0 {
                // Every byte of a buffer reads zero.
                // In the block, which nothing else holds.
                unsafe { std::ptr::write_bytes(current.block.base().as_ptr(), 0, current.used as usize) };
                current.used = 0;
            }
            let offset = current.used.next_multiple_of(alignment);
            if offset + bytes.max(1) <= current.block.bytes() {
                current.used = offset + bytes;
                return Ok(PageRange { block: Some(current.block.clone()), offset, bytes });
            }
        }
        let pages = bytes.max(1).div_ceil(page).max(CARVED_BLOCK_PAGES);
        let block = PageBlock::allocate(pages * page, entry)?;
        let room = |held: &Filling| held.block.bytes() - held.used;
        let fresh = Filling { block: block.clone(), used: bytes };
        let first = match filling.get(&key) {
            Some(current) if !current.idle() && room(current) >= room(&fresh) => false,
            current => {
                let first = current.is_none();
                filling.insert(key, fresh);
                first
            }
        };
        drop(filling);
        // Run at once where the holder is gone already: the range below
        // keeps the block, which returns as it drops ([`Self::returned`]).
        if first {
            entry.on_close(Box::new(move || Self::process().closed(key)));
        }
        Ok(PageRange { block: Some(block), offset: 0, bytes })
    }
}

/// `len` bytes at `pointer` that `_owner` keeps alive.
struct BorrowedRegion {
    pointer: NonNull<u8>,
    len: u64,
    _owner: Arc<dyn std::any::Any + Send + Sync>,
}

// The bytes are the owner's, which is `Send + Sync`; every access goes
// through the raw address under the submission discipline documented at the
// module level, as for the heap's.
unsafe impl Send for BorrowedRegion {}
unsafe impl Sync for BorrowedRegion {}

/// One host allocation, shared by handle: what device work binds and the
/// host reads. The host writes only a [`WritableBuffer`], the form of the
/// memory this crate allocates; a buffer over borrowed read-only memory
/// ([`Buffer::borrowed`]) is never one, so it has no host write path.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<Allocation>,
}

/// A [`Buffer`] over memory this crate allocated, which the host also
/// writes.
#[derive(Clone)]
pub struct WritableBuffer(Buffer);

impl WritableBuffer {
    /// `bytes` at `alignment` (a power of two, at most the host page), and
    /// at least [`LEAST_ALIGNMENT`]: host pages of its own, every one backed and zero before it returns
    /// and charged whole to `entry` before they are allocated: the bytes in
    /// whole host pages, at least one, so an empty allocation still has a
    /// live address for typed native views, or, where they are at most
    /// [`CARVED_LIMIT`], a range of the entry's filling block ([`Carver`]). Returned to
    /// the OS when the last handle over the pages drops. Nothing is wired:
    /// the OS may compress or page it as any memory of the process.
    ///
    /// # Panics
    /// Where `alignment` is not a power of two dividing the host page: no
    /// profile asks for more than a page.
    pub fn new(bytes: u64, alignment: u64, entry: &Arc<dyn Entry>) -> Result<Self, ExecutionError> {
        let page = seismic_memory::host_page().map_err(allocation_error)?;
        assert!(
            alignment.is_power_of_two() && page % alignment == 0,
            "invariant violated: a CPU allocation aligned to {alignment} bytes, which a {page}-byte host page is not"
        );
        if bytes > CARVED_LIMIT {
            return Self::pages(bytes, entry);
        }
        let range = Carver::process().carve(bytes, alignment.max(LEAST_ALIGNMENT), page, entry)?;
        Ok(Self(Buffer { inner: Arc::new(Allocation::Pages(range)) }))
    }

    /// `bytes` in whole host pages of its own, at least one, never carved:
    /// at the page, every one backed and zero, charged whole to `entry`
    /// before they are allocated and returned to the OS as the last handle
    /// drops. What a buffer needing the page's alignment takes (a granule
    /// pool's block), whatever its size.
    pub fn pages(bytes: u64, entry: &Arc<dyn Entry>) -> Result<Self, ExecutionError> {
        let page = seismic_memory::host_page().map_err(allocation_error)?;
        let whole = bytes
            .max(1)
            .checked_next_multiple_of(page)
            .ok_or_else(|| allocation_error(AllocationFailure::Unrepresentable))?;
        let range = PageRange { block: Some(PageBlock::allocate(whole, entry)?), offset: 0, bytes };
        Ok(Self(Buffer { inner: Arc::new(Allocation::Pages(range)) }))
    }

    /// Copies `bytes` into the buffer at `offset`. The range lies inside the
    /// allocation: the runtime sizes every transfer from the schema, so an
    /// out-of-range transfer is a violated precondition of this wrapper
    /// (§13.3.3).
    pub fn write(&self, offset: u64, bytes: &[u8]) {
        let range = self.range(offset, bytes.len());
        // In-range by `range`; the source is a separate host slice.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data_pointer().add(range), bytes.len())
        };
    }
}

impl std::ops::Deref for WritableBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        &self.0
    }
}

impl std::borrow::Borrow<Buffer> for WritableBuffer {
    fn borrow(&self) -> &Buffer {
        &self.0
    }
}

impl Buffer {
    /// A buffer over `len` bytes of host memory at `pointer`, which `owner`
    /// holds: no bytes are copied or allocated, and the host only reads
    /// them.
    ///
    /// # Safety
    /// The bytes are readable for as long as `owner` lives, and nothing
    /// writes the ones a launch binding the buffer reads while it runs.
    pub unsafe fn borrowed(pointer: NonNull<u8>, len: u64, owner: Arc<dyn std::any::Any + Send + Sync>) -> Self {
        Self {
            inner: Arc::new(Allocation::Borrowed(BorrowedRegion { pointer, len, _owner: owner })),
        }
    }

    /// Logical bytes.
    pub fn len(&self) -> u64 {
        match &*self.inner {
            Allocation::Pages(range) => range.bytes,
            Allocation::Borrowed(region) => region.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The host address of byte 0. Valid while any handle is alive.
    pub fn data_pointer(&self) -> *mut u8 {
        match &*self.inner {
            // In the block by construction.
            Allocation::Pages(range) => unsafe { range.block().base().as_ptr().add(range.offset as usize) },
            Allocation::Borrowed(region) => region.pointer.as_ptr(),
        }
    }

    /// Copies bytes out of the buffer at `offset` (same precondition as
    /// [`WritableBuffer::write`]).
    pub fn read(&self, offset: u64, into: &mut [u8]) {
        let range = self.range(offset, into.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data_pointer().add(range),
                into.as_mut_ptr(),
                into.len(),
            )
        };
    }

    fn range(&self, offset: u64, len: usize) -> usize {
        let end = offset.checked_add(len as u64);
        match end {
            Some(end) if end <= self.len() => offset as usize,
            _ => panic!(
                "Buffer precondition violated: transfer of {len} bytes at offset {offset} exceeds the {}-byte allocation",
                self.len()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Buffer, Entry, ExecutionError, WritableBuffer, CARVED_BLOCK_PAGES, CARVED_LIMIT, LEAST_ALIGNMENT};
    use seismic_memory::entry::{Change, Refused, Tally, Wiring};
    use seismic_topology::StoreIdentity;
    use std::sync::Arc;
    use std::sync::Mutex;

    /// A tally whose holder can go: what a claim is to the ledger's entry.
    struct Held {
        tally: Tally,
        closing: Mutex<Option<Vec<Box<dyn FnOnce() + Send>>>>,
    }

    impl Held {
        fn new() -> (Arc<Self>, Tally) {
            let tally = Tally::unbounded(StoreIdentity::HostRam);
            (Arc::new(Self { tally: tally.clone(), closing: Mutex::new(Some(Vec::new())) }), tally)
        }

        /// The holder is gone.
        fn close(&self) {
            let closing = self.closing.lock().unwrap().take().expect("closed once");
            closing.into_iter().for_each(|closed| closed());
        }
    }

    impl Entry for Held {
        fn store(&self) -> &StoreIdentity {
            self.tally.store()
        }

        fn change(&self) -> Change {
            self.tally.change()
        }

        fn on_close(&self, closed: Box<dyn FnOnce() + Send>) {
            match self.closing.lock().unwrap().as_mut() {
                Some(closing) => closing.push(closed),
                None => closed(),
            }
        }

        fn closed(&self) -> bool {
            self.closing.lock().unwrap().is_none()
        }

        fn take(&self, change: &Change, bytes: u64, wiring: Wiring) -> Result<(), Refused> {
            self.tally.take(change, bytes, wiring)
        }

        fn took(&self, change: &Change, bytes: u64, wiring: Wiring) {
            self.tally.took(change, bytes, wiring);
        }

        fn freed(&self, change: &Change, bytes: u64, wiring: Wiring) {
            self.tally.freed(change, bytes, wiring);
        }
    }

    /// The block an entry keeps idle returns as its holder goes, inside
    /// that call, not as an unrelated carve finds it.
    #[test]
    fn an_idle_block_returns_as_the_holder_goes() {
        let (held, tally) = Held::new();
        let entry: Arc<dyn Entry> = held.clone();
        drop(WritableBuffer::new(64, 64, &entry).expect("buffer"));
        assert!(tally.held() > 0, "the idle block is kept for the next buffer");
        held.close();
        assert_eq!(tally.held(), 0, "the holder's going returned it");
    }

    /// A block with a buffer in it returns as that buffer drops once the
    /// holder is gone, inside that free.
    #[test]
    fn a_block_returns_as_its_last_buffer_drops_after_the_holder_went() {
        let (held, tally) = Held::new();
        let entry: Arc<dyn Entry> = held.clone();
        let first = WritableBuffer::new(64, 64, &entry).expect("buffer");
        let second = WritableBuffer::new(64, 64, &entry).expect("buffer");
        held.close();
        assert!(tally.held() > 0, "a buffer still lies in the block");
        drop(first);
        assert!(tally.held() > 0, "a buffer still lies in the block");
        drop(second);
        assert_eq!(tally.held(), 0, "the last buffer's free returned the block");
    }

    /// An entry whose holder is alive: it keeps its filling block.
    fn entry() -> Arc<dyn Entry> {
        Held::new().0
    }

    /// An empty allocation still has a live address, at its alignment, in
    /// pages the entry is charged.
    #[test]
    fn zero_logical_bytes_have_live_aligned_backing() {
        let page = seismic_memory::host_page().unwrap();
        let tally = Tally::unbounded(StoreIdentity::HostRam);
        let entry: Arc<dyn Entry> = Arc::new(tally.clone());
        let buffer = WritableBuffer::new(0, 64, &entry).expect("empty buffer");
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        let pointer = buffer.data_pointer() as usize;
        assert_ne!(pointer, 0);
        assert_eq!(pointer % 64, 0);
        assert_eq!(tally.held(), CARVED_BLOCK_PAGES * page);
    }

    /// Sub-page requests lie one after another, at the least alignment
    /// whatever they ask, and disjoint, in the one block the entry's charge is whole pages of; a
    /// request of a page or more takes whole pages of its own, and the
    /// pages return as the last buffer over them drops.
    #[test]
    fn sub_page_requests_share_one_charged_block() {
        let page = seismic_memory::host_page().unwrap();
        let tally = Tally::unbounded(StoreIdentity::HostRam);
        let entry: Arc<dyn Entry> = Arc::new(tally.clone());
        let block = CARVED_BLOCK_PAGES * page;
        let sizes = [1, 8, 100, page / 2, page - 1];
        let mut buffers = Vec::new();
        for &bytes in &sizes {
            let buffer = WritableBuffer::new(bytes, 1, &entry).expect("buffer");
            assert_eq!(buffer.len(), bytes);
            assert_eq!(buffer.data_pointer() as u64 % LEAST_ALIGNMENT, 0, "a {bytes}-byte buffer is at the least alignment");
            assert_eq!(tally.held(), block, "a {bytes}-byte buffer shares the entry's one block");
            buffers.push(buffer);
        }
        for (earlier, later) in buffers.iter().zip(&buffers[1..]) {
            assert!(
                earlier.data_pointer() as u64 + earlier.len() <= later.data_pointer() as u64,
                "carved buffers are disjoint, in order"
            );
        }
        let whole = WritableBuffer::new(CARVED_LIMIT + 1, 8, &entry).expect("a large buffer");
        assert_eq!(whole.data_pointer() as u64 % page, 0, "a large buffer starts its own page");
        assert_eq!(tally.held(), block + CARVED_LIMIT + page, "a large buffer is charged whole pages of its own");
        drop(whole);
        assert_eq!(tally.held(), block);
        drop(buffers);
        assert_eq!(tally.held(), 0, "a tally has no holder: its block returns as its last buffer drops");
    }

    /// A full block is replaced: the next buffer lies in a new one, and
    /// each returns as its own buffers drop.
    #[test]
    fn a_full_block_is_replaced() {
        let page = seismic_memory::host_page().unwrap();
        let tally = Tally::unbounded(StoreIdentity::HostRam);
        let entry: Arc<dyn Entry> = Arc::new(tally.clone());
        let block = CARVED_BLOCK_PAGES * page;
        let first = WritableBuffer::new(page - 1, 1, &entry).expect("buffer");
        let filler = (1..CARVED_BLOCK_PAGES)
            .map(|_| WritableBuffer::new(page - 1, 1, &entry).expect("buffer"))
            .collect::<Vec<_>>();
        assert_eq!(tally.held(), block);
        let next = WritableBuffer::new(page - 1, 1, &entry).expect("buffer");
        assert_eq!(tally.held(), 2 * block);
        drop((first, filler));
        assert_eq!(tally.held(), block, "the full block returns with its buffers");
        drop(next);
        assert_eq!(tally.held(), 0, "a tally has no holder: its filling block returns with its last buffer");
    }

    /// A buffer made and dropped with no sibling alive reuses the entry's
    /// block: the same pages, zero again, never a new block.
    #[test]
    fn a_lone_buffer_reuses_the_entrys_block_zeroed() {
        let (held, _tally) = Held::new();
        let entry: Arc<dyn Entry> = held;
        let first = WritableBuffer::new(64, 8, &entry).expect("buffer");
        let address = first.data_pointer();
        first.write(0, &[7; 64]);
        drop(first);
        let second = WritableBuffer::new(64, 8, &entry).expect("buffer");
        assert_eq!(second.data_pointer(), address, "the idle block is reused");
        let mut bytes = [1; 64];
        second.read(0, &mut bytes);
        assert!(bytes.iter().all(|byte| *byte == 0), "a reused buffer reads zero");
    }

    /// Made, written and dropped, per buffer, against the heap (a
    /// microbench, run once; prints figures): alone (no sibling alive) and
    /// beside a live sibling.
    #[test]
    fn a_made_written_dropped_buffer_costs_per_buffer_against_the_heap() {
        use std::time::Instant;
        let measure = |bytes: u64, make: &dyn Fn()| {
            let rounds = ((256u64 << 20) / bytes).clamp(200, 20_000);
            for _ in 0..rounds.min(200) {
                make();
            }
            let start = Instant::now();
            for _ in 0..rounds {
                make();
            }
            start.elapsed().as_nanos() as f64 / rounds as f64
        };
        for sibling in [false, true] {
            for bytes in [64u64, 4096, 65536, 1 << 20] {
                let entry = entry();
                let _sibling = sibling.then(|| WritableBuffer::new(8, 8, &entry).expect("sibling"));
                let data = vec![1u8; bytes as usize];
                let heap = measure(bytes, &|| {
                    let mut buffer = vec![0u8; bytes as usize];
                    buffer.copy_from_slice(&data);
                    std::hint::black_box(&buffer);
                });
                let carved = measure(bytes, &|| {
                    let buffer = WritableBuffer::new(bytes, 8, &entry).expect("buffer");
                    buffer.write(0, &data);
                    std::hint::black_box(&buffer);
                });
                println!(
                    "{bytes:>8} B sibling={sibling:<5}: {carved:>9.0} ns per buffer, heap {heap:>8.0} ns, {:.1}x",
                    carved / heap
                );
            }
        }
    }


    /// A borrowed buffer reads its owner's bytes in place; it has no host
    /// write path, which only a [`WritableBuffer`] offers.
    #[test]
    fn a_borrowed_buffer_reads_its_owners_bytes() {
        let owner = std::sync::Arc::new((0..16u8).collect::<Vec<_>>());
        let pointer = std::ptr::NonNull::new(owner.as_ptr().cast_mut()).unwrap();
        let buffer = unsafe { Buffer::borrowed(pointer, 16, owner.clone()) };
        assert_eq!(buffer.len(), 16);
        assert_eq!(buffer.data_pointer().cast_const(), owner.as_ptr());
        let mut bytes = [0; 4];
        buffer.read(12, &mut bytes);
        assert_eq!(bytes, [12, 13, 14, 15]);
    }

    #[test]
    fn empty_transfers_are_bounded_by_logical_length() {
        let buffer = WritableBuffer::new(0, 64, &entry()).expect("empty buffer");
        buffer.write(0, &[]);
        let mut bytes = [];
        buffer.read(0, &mut bytes);
    }

    /// A large buffer is its own whole host pages, zero, charged at their
    /// bytes (never the allocator's layout, whose pages a free would keep)
    /// until its last handle drops; a refused charge allocates nothing.
    #[test]
    fn a_buffer_is_charged_its_pages_until_dropped() {
        let page = seismic_memory::host_page().unwrap();
        let whole = CARVED_LIMIT + page;
        let tally = Tally::limited(StoreIdentity::HostRam, 3 * whole);
        let entry: Arc<dyn Entry> = Arc::new(tally.clone());
        let block = WritableBuffer::new(CARVED_LIMIT + 1, 64, &entry).expect("buffer");
        assert_eq!(block.len(), CARVED_LIMIT + 1);
        assert_eq!(block.data_pointer() as u64 % page, 0);
        assert_eq!(tally.held(), whole);
        let mut bytes = vec![1; (CARVED_LIMIT + 1) as usize];
        block.read(0, &mut bytes);
        assert!(bytes.iter().all(|byte| *byte == 0), "a buffer reads zero");
        assert!(matches!(
            WritableBuffer::new(3 * CARVED_LIMIT, 64, &entry),
            Err(ExecutionError::AllocationCapacity { available, .. }) if available == 2 * whole
        ));
        let handle = Buffer::clone(&block);
        drop(block);
        assert_eq!(tally.held(), whole, "a live handle keeps the block");
        drop(handle);
        assert_eq!(tally.held(), 0);
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.len())
            .finish()
    }
}
