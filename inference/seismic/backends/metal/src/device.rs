//! The Metal device service: one `MTLDevice`, one command queue, shared
//! storage-mode buffers (spec §12.3, R9).
//!
//! The service allocates, writes, reads and measures buffers. It infers
//! nothing about what a buffer holds.

use crate::Metal;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions,
};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::DeviceService;
use seismic_lang::registry::{chunk_entries, record_table_bytes, record_table_entry, RECORD_TABLE_ENTRY_BYTES};
use std::fmt;
use seismic_reservation::{Backing, Forming, HostRange, PageSize, Rebinding, Reserved, Resizer};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;
use std::{any::Any, ptr::NonNull};

/// A retained `MTLDevice`. `MTLDevice` is `Send + Sync`; identity is the
/// device's registry id.
#[derive(Clone)]
pub struct DeviceHandle {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

// Metal resource objects are explicitly thread-safe; objc2's protocol-object
// erasure cannot currently express those protocol-level guarantees.
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}

impl DeviceHandle {
    /// Cheap enumeration of unopened physical Metal devices. This performs no
    /// queue creation, native compilation, or profiling.
    pub fn discover() -> Vec<Self> {
        crate::pooled(|| {
            MTLCopyAllDevices()
                .into_iter()
                .map(|device| Self { device })
                .collect()
        })
    }

    /// The system default Metal device.
    pub fn system_default() -> Result<Self, TargetError> {
        let device = crate::pooled(|| MTLCreateSystemDefaultDevice())
            .ok_or_else(|| TargetError::DeviceUnavailable("no Metal device".into()))?;
        Ok(Self { device })
    }

    pub fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub fn name(&self) -> String {
        crate::pooled(|| self.device.name().to_string())
    }

    /// `MTLDevice.maxBufferLength`: the single-allocation limit. It is neither
    /// memory capacity nor availability.
    pub fn max_allocation_bytes(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    /// `MTLDevice.hasUnifiedMemory`: CPU and GPU share one memory. Whether
    /// that memory is the whole host RAM pool is a platform qualification,
    /// not implied by this flag alone.
    pub fn has_unified_memory(&self) -> bool {
        self.device.hasUnifiedMemory()
    }

    /// `MTLDevice.recommendedMaxWorkingSetSize`: Metal's advisory ceiling for
    /// this device's working set. Advice, not physical capacity.
    pub fn recommended_working_set_bytes(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// `MTLDevice.currentAllocatedSize`: bytes of resources this process has
    /// allocated on the device, including resources Seismic does not own.
    /// Each buffer counts its whole length (a reservation's chunks each
    /// once); the device holds the working set to that count (its command
    /// buffers fail with out-of-memory beyond the recommended working set,
    /// M4 Pro, 2026-10-27).
    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

impl PartialEq for DeviceHandle {
    fn eq(&self, other: &Self) -> bool {
        self.registry_id() == other.registry_id()
    }
}

impl fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHandle")
            .field("name", &self.name())
            .field("registry_id", &self.registry_id())
            .finish()
    }
}

/// macOS lets a residency set's requested residency lapse about 1 s after
/// the set's last GPU work and un-wires its allocations until the next GPU
/// work wires them again, about 10 ms per GB before that work's first result
/// (M1 Ultra, macOS 26.7: 26 GB un-wired 1.05 s after a request, +0.26 s to
/// the next first token). Renewing ten times inside that second keeps them
/// wired through every idle gap; stopping wakes the renewer at once, so the
/// interval adds no latency to an unload or exit.
const RESIDENCY_RENEWAL: Duration = Duration::from_millis(100);

/// The device's residency set and its renewal. While the set holds an
/// allocation one thread renews the set's residency every
/// `RESIDENCY_RENEWAL`; the release of its last allocation stops and joins
/// that thread, and the next allocation starts another. Wired memory is
/// released by releasing allocations, never by letting residency lapse.
struct Residency {
    wired: Mutex<Wired>,
    stop: Condvar,
}

struct Wired {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    renewer: Option<JoinHandle<()>>,
    generation: u64,
    renewals: u64,
}

// The set is used only under the residency mutex, on whichever thread holds
// it.
unsafe impl Send for Wired {}

impl Residency {
    fn new(set: Retained<ProtocolObject<dyn MTLResidencySet>>) -> Arc<Self> {
        Arc::new(Self {
            wired: Mutex::new(Wired {
                set,
                renewer: None,
                generation: 0,
                renewals: 0,
            }),
            stop: Condvar::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Wired> {
        self.wired.lock().expect("Metal residency lock poisoned")
    }

    /// Add `buffers` to the set and wire them now, on the calling thread, in
    /// one commit. A buffer the set only holds is wired by the first GPU
    /// work that binds it, on the thread submitting that work (a 20 GiB
    /// buffer: +142 ms before its first read, M4 Pro, 2026-10-25), which a
    /// reservation's resize prepares away from.
    fn add(self: &Arc<Self>, buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>]) {
        if buffers.is_empty() {
            return;
        }
        let mut wired = self.lock();
        crate::pooled(|| {
            for buffer in buffers {
                wired.set.addAllocation(ProtocolObject::from_ref(&**buffer));
            }
            wired.set.commit();
            wired.set.requestResidency();
        });
        if wired.renewer.is_none() {
            let residency = self.clone();
            let generation = wired.generation;
            let renewer = std::thread::Builder::new()
                .name("metal-residency".into())
                .spawn(move || residency.renew(generation))
                .expect("could not start the Metal residency renewer");
            wired.renewer = Some(renewer);
        }
    }

    /// Remove `buffers` from the set in one commit.
    fn remove(&self, buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>]) {
        if buffers.is_empty() {
            return;
        }
        let renewer = {
            let mut wired = self.lock();
            crate::pooled(|| {
                for buffer in buffers {
                    wired.set.removeAllocation(ProtocolObject::from_ref(&**buffer));
                }
                wired.set.commit();
            });
            if wired.set.allocationCount() > 0 {
                return;
            }
            wired.generation += 1;
            wired.renewer.take()
        };
        self.stop.notify_all();
        if let Some(renewer) = renewer {
            renewer.join().expect("the Metal residency renewer panicked");
        }
    }

    /// Renews residency until `generation` is stopped. The first renewal is
    /// immediate: a request made before the set held anything may already
    /// have lapsed.
    fn renew(&self, generation: u64) {
        let mut wired = self.lock();
        while wired.generation == generation {
            crate::pooled(|| wired.set.requestResidency());
            wired.renewals += 1;
            wired = self
                .stop
                .wait_timeout(wired, RESIDENCY_RENEWAL)
                .expect("Metal residency lock poisoned")
                .0;
        }
    }
}

/// Buffers' membership of their device's residency set. Each is wired from
/// the commit that adds it until it leaves (dropped, or split off and that
/// part dropped), so the kernel never compresses or swaps it while macOS
/// honors the set's renewed residency.
pub struct WiredAllocation {
    residency: Arc<Residency>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

unsafe impl Send for WiredAllocation {}
unsafe impl Sync for WiredAllocation {}

impl WiredAllocation {
    fn wire(
        residency: &Arc<Residency>,
        buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> Self {
        residency.add(&buffers);
        Self {
            residency: residency.clone(),
            buffers,
        }
    }

    /// Join `other`'s buffers, which stay wired.
    fn append(&mut self, mut other: Self) {
        self.buffers.append(&mut other.buffers);
    }

    /// The buffers from `at` on, which stay wired until it is dropped.
    fn split_off(&mut self, at: usize) -> Self {
        Self {
            residency: self.residency.clone(),
            buffers: self.buffers.split_off(at),
        }
    }
}

impl Drop for WiredAllocation {
    fn drop(&mut self) {
        self.residency.remove(&self.buffers);
    }
}

/// What a direct launch binds at one buffer index, from an offset
/// ([`crate::DirectLaunch::buffers`]).
#[derive(Clone, Copy)]
pub enum Binding<'a> {
    Buffer(&'a MetalBuffer),
    /// A reservation's buffer over its whole range, which exists only while
    /// every granule is backed ([`ReservedBuffer::bind`]).
    Reserved(&'a ReservedBuffer),
}

impl Binding<'_> {
    /// `bind` given the Metal buffer this binds, for as long as `bind` runs.
    pub(crate) fn with<R>(self, bind: impl FnOnce(&ProtocolObject<dyn MTLBuffer>) -> R) -> R {
        match self {
            Self::Buffer(buffer) => bind(buffer.raw()),
            Self::Reserved(reserved) => reserved.bind(bind),
        }
    }
}

/// One shared-storage Metal buffer of fixed length: allocated by Metal
/// (wired), or wrapping a read-only host mapping. Device work binds it
/// ([`MetalBuffer::raw`]) and the host reads it; the host writes only a
/// [`WritableMetalBuffer`], the form of a buffer Metal allocated.
#[derive(Clone)]
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: u64,
    _host_owner: Option<Arc<dyn Any + Send + Sync>>,
    /// Absent for a read-only host mapping, whose pages stay file-backed
    /// and outside the residency set: resident only where it is bound, so
    /// no record table addresses it.
    _wired: Option<Arc<WiredAllocation>>,
}

/// A [`MetalBuffer`] Metal allocated, which the host also writes.
#[derive(Clone)]
pub struct WritableMetalBuffer(MetalBuffer);

impl WritableMetalBuffer {
    pub(crate) fn write_bytes(&self, offset: u64, bytes: &[u8]) {
        // Shared storage Metal allocated: host-writable for its lifetime.
        unsafe { self.0.write_device_bytes(offset, bytes) }
    }
}

impl std::ops::Deref for WritableMetalBuffer {
    type Target = MetalBuffer;
    fn deref(&self) -> &MetalBuffer {
        &self.0
    }
}

impl std::borrow::Borrow<MetalBuffer> for WritableMetalBuffer {
    fn borrow(&self) -> &MetalBuffer {
        &self.0
    }
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    /// The Metal buffer device work binds.
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// The GPU virtual address of byte `offset`, held by slab and record
    /// tables (macOS 15 and later).
    pub fn gpu_address(&self, offset: u64) -> u64 {
        self.buffer.gpuAddress() + offset
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The host address of byte 0: shared storage mode's `contents()`,
    /// host-visible for the buffer's whole lifetime.
    fn host_base(&self) -> *mut u8 {
        self.buffer.contents().as_ptr().cast::<u8>()
    }

    /// Write `bytes` at `offset` as device work does: the bound executor's
    /// own inputs, under the schedule's access plan.
    ///
    /// # Safety
    /// The schedule plans these bytes as written by this work, so the
    /// buffer is one Metal allocated, never a read-only host mapping.
    pub(crate) unsafe fn write_device_bytes(&self, offset: u64, bytes: &[u8]) {
        unsafe { host_write(self.host_base(), self.len, offset, bytes) }
    }

    pub(crate) fn read_bytes(&self, offset: u64, into: &mut [u8]) {
        unsafe { host_read(self.host_base(), self.len, offset, into) }
    }
}

/// The committed prefix of a host address reservation, resized in place
/// ([`MetalResizer::prepare`]): Metal buffers over its backed granules
/// ([`GranuleView`]), which device work reaches through the reservation's
/// record table (a [`MetalBuffer`]), and while every granule is backed also
/// binds directly ([`Binding::Reserved`]). The host addresses the range in
/// place.
#[derive(Clone)]
pub struct ReservedBuffer {
    reservation: Arc<Reservation>,
}

impl ReservedBuffer {
    /// Committed bytes.
    pub fn len(&self) -> u64 {
        self.reservation.state().bytes
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The host address of byte 0: the range's base, which never moves.
    pub fn host_base(&self) -> NonNull<u8> {
        self.reservation.backing().range.base()
    }

    /// The record table device work reaches the records through, which the
    /// reservation formed with it and keeps as its chunks change.
    pub fn table(&self) -> WritableMetalBuffer {
        self.reservation.backing().table.clone()
    }

    /// The GPU virtual address of byte `offset`, in the buffer holding it,
    /// which must be backed; its address is fixed until the view changes
    /// form ([`ChunkChange::Replace`]).
    pub fn gpu_address(&self, offset: u64) -> u64 {
        let granule = self.reservation.backing().range.granule() as u64;
        match &self.reservation.state().view {
            GranuleView::Chunks(chunks) => chunks.buffers[(offset / granule) as usize].gpuAddress() + offset % granule,
            GranuleView::Whole(whole) => whole.buffers[0].gpuAddress() + offset,
        }
    }

    /// Whether device work binds it directly now: its view is the whole
    /// form ([`GranuleView::Whole`]), which a form chosen at a resize
    /// decides, not its backed granules alone.
    pub fn is_whole(&self) -> bool {
        matches!(self.reservation.state().view, GranuleView::Whole(_))
    }

    /// `bind` given the one buffer over the whole range
    /// ([`GranuleView::Whole`]), under the reservation's lock, so no shrink
    /// retires it meanwhile; work bound to it runs before any shrink's
    /// release, as shrinks run only between flights.
    ///
    /// # Panics
    /// Unless every granule is backed: only then is there one, and only then
    /// does device work bind the reservation directly.
    fn bind<R>(&self, bind: impl FnOnce(&ProtocolObject<dyn MTLBuffer>) -> R) -> R {
        match &self.reservation.state().view {
            GranuleView::Whole(whole) => bind(&whole.buffers[0]),
            GranuleView::Chunks(_) => {
                panic!("device work binds a reservation directly only while every granule is backed")
            }
        }
    }

    /// Write `bytes` at `offset` of the committed bytes.
    pub fn write_bytes(&self, offset: u64, bytes: &[u8]) {
        unsafe { host_write(self.host_base().as_ptr(), self.len(), offset, bytes) }
    }

    /// Read committed bytes from `offset` into `into`.
    pub fn read_bytes(&self, offset: u64, into: &mut [u8]) {
        unsafe { host_read(self.host_base().as_ptr(), self.len(), offset, into) }
    }
}

/// The host address of `[offset, offset + len)` of `length` bytes at
/// `base`. Precondition (FFI wrapper): the range lies inside them.
fn host_range(base: *mut u8, length: u64, offset: u64, len: usize) -> *mut u8 {
    let end = offset.checked_add(len as u64).filter(|end| *end <= length);
    assert!(
        end.is_some(),
        "Metal host access [{offset}, +{len}) exceeds the {length}-byte buffer"
    );
    unsafe { base.add(offset as usize) }
}

/// # Safety
/// `base` addresses `length` writable host bytes.
unsafe fn host_write(base: *mut u8, length: u64, offset: u64, bytes: &[u8]) {
    let destination = host_range(base, length, offset, bytes.len());
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len()) };
}

/// # Safety
/// `base` addresses `length` readable host bytes.
unsafe fn host_read(base: *mut u8, length: u64, offset: u64, into: &mut [u8]) {
    let source = host_range(base, length, offset, into.len());
    unsafe { std::ptr::copy_nonoverlapping(source, into.as_mut_ptr(), into.len()) };
}

/// The least chunk in which a reservation backs its committed bytes. Each
/// chunk is its own VM object (a named memory entry), so releasing one frees
/// its pages at once: a VM object shared by kept and released bytes would
/// keep the released pages while Metal still references the object.
///
/// A chunk costs a fixed ~4.9 us to form (its memory entry and mapping, its
/// no-copy buffer and table entries; the residency set's commit is under
/// 0.1 us of it) beside ~34 us per MiB to back its bytes, so the granule is
/// the least power-of-two chunk whose fixed cost is at most 10% of its
/// bytes': 2 MiB (7%; 1 MiB would be 14%). Backing 1 GiB at once
/// (`reservation_chunk_cost`, M4 Pro):
/// 355 ms in 16 KiB chunks, 54 ms in 256 KiB, 40 ms in 1 MiB,
/// 37.6 ms in 2 MiB, 35.4 ms in 32 MiB; a growth by one chunk, 5.8 us at
/// 16 KiB and 66 us at 2 MiB. GPU reads through the record table took
/// 4.4-5.7 ms a GiB at every chunk size, so the chunk count costs device
/// work nothing.
pub const RESERVATION_GRANULE: usize = 2 << 20;

/// Whether the host addresses a reservation's backed bytes in place: its
/// buffer wraps the host range without copying (`newBufferWithBytesNoCopy`).
pub const RESERVATIONS_HOST_ADDRESSED: bool = true;

/// What device work reaches a reservation's backed bytes through chunks of:
/// each chunk is its own buffer (a buffer GPU work touches is wired whole,
/// so one buffer over the range would hold every page a release frees; M4
/// Pro, macOS 15.7, 2026-10-27) of whole [`CHUNK_UNIT`]s. While every
/// granule is backed one buffer over the range takes the chunks' place
/// ([`GranuleView::Whole`]); a shrink replaces it with the chunks it keeps.
pub const RESERVATION_CHUNKS: Option<u64> = Some(CHUNK_UNIT);

/// 16 KiB, the largest macOS page: every chunk's no-copy buffer starts and
/// ends on a host page.
const CHUNK_UNIT: u64 = 16 << 10;

/// Metal's backing of a host reservation (`seismic_reservation`):
/// granules of anonymous memory, each its own VM object, the Metal buffers
/// over the backed ones ([`GranuleView`]), and the reservation's record
/// table.
///
/// GPU work that touches a buffer wires all of it, so one buffer over a
/// partly backed range would hold every page a shrink frees: while a
/// granule is unbacked each backed granule is wrapped once
/// (`newBufferWithBytesNoCopy`) as its own buffer (a chunk) and joins the
/// residency set, so only committed chunks are wired; device work reaches
/// records through the table, entry `r` the device address of record `r` in
/// its buffer (no record crosses a chunk). A growth forms only its new
/// chunks; a shrink retires its tail chunks, dropped (leaving the set) by
/// its release before their granules are unmapped, so device work never
/// reads a page remapped under a live buffer.
///
/// Metal counts each buffer's whole length in `currentAllocatedSize`, which
/// the device's grant is read from, and the reservation is charged its
/// backing once: so each backed byte is in exactly one buffer. While every
/// granule is backed that is one buffer over the range in the chunks'
/// place, which device work also binds directly; a change between the two
/// forms ([`ChunkChange::Replace`]) forms the next form's buffers beside
/// device work and rewrites every table entry as it is applied.
pub struct Granules {
    /// The host address range the granules are of.
    range: HostRange,
    device: DeviceHandle,
    residency: Arc<Residency>,
    /// The record table, whose entries for a buffer's records are written
    /// as device work comes to reach them through it: entries of released
    /// granules are never read.
    table: WritableMetalBuffer,
    /// Bytes of one record, a divisor of the granule.
    record: usize,
}

unsafe impl Send for Granules {}
unsafe impl Sync for Granules {}

/// A change of a reservation's buffers.
pub enum ChunkChange {
    /// A growth of chunks: its new chunks, their records' table entries
    /// written, which no device work reads before the growth is installed.
    Grow(WiredAllocation),
    /// A shrink of chunks: the chunks it keeps.
    Shrink(usize),
    /// A change of form formed beside the view, a growth to the whole form
    /// or a shrink from it into chunks ([`ViewForm::Whole`],
    /// [`ViewForm::Chunks`]): the view that takes the installed one's place,
    /// its buffers formed and wired, whose table entries its apply writes
    /// over every record it spans.
    Replace(GranuleView),
    /// A shrink from the whole form into chunks over its leading granules,
    /// formed by its apply in that form's place
    /// (`seismic_reservation::Forming::InPlace`).
    InPlace(usize),
}

/// The form of a reservation's view ([`GranuleView`]), which the
/// reservation's holder chooses (`seismic_reservation::Backing::Form`): the
/// buffers of a change between forms formed beside the view's are counted
/// with them until its release, so the holder takes the next form's bytes
/// once more for them, or takes the whole form not at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewForm {
    /// A chunk over each backed granule.
    #[default]
    Chunks,
    /// One buffer over the range: only over every granule, within the
    /// device's longest buffer (`seismic_topology::Reservation::binds_whole`).
    Whole,
}

/// The Metal buffers over a reservation's backed granules, every one in the
/// device's residency set, each backed byte in exactly one of them.
pub enum GranuleView {
    /// A chunk over each backed granule, in order.
    Chunks(WiredAllocation),
    /// One buffer over the range, every granule backed
    /// ([`ViewForm::Whole`]), which device work also binds directly
    /// ([`ReservedBuffer::bind`]).
    Whole(WiredAllocation),
}

type Reservation = Reserved<Granules>;
pub type MetalResizer = Resizer<Granules>;

impl Granules {
    /// Whether the view over the first `granules` granules may be one
    /// buffer over the range: they are every one, and lie within the
    /// device's longest buffer.
    fn whole_at(&self, granules: usize) -> bool {
        granules > 0
            && granules == self.range.granules()
            && (self.range.granule() * granules) as u64 <= self.device.max_allocation_bytes()
    }

    /// The form a reservation formed with its first `granules` granules
    /// backed takes: whole wherever it may be, as forming it holds nothing
    /// beside it.
    fn formed_at(&self, granules: usize) -> ViewForm {
        match self.whole_at(granules) {
            true => ViewForm::Whole,
            false => ViewForm::Chunks,
        }
    }

    /// A no-copy buffer over granules `[from, to)`.
    fn buffer(&self, from: usize, to: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, ExecutionError> {
        let bytes = self.range.granule() * (to - from);
        crate::pooled(|| unsafe {
            self.device.raw().newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new_unchecked(self.range.at(from)).cast(),
                bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        })
        // As for an allocated buffer, Metal states no reason for a buffer it
        // does not create: the device's memory refused it.
        .ok_or(ExecutionError::AllocationCapacity {
            required: (bytes as u64).into(),
            available: 0,
        })
    }

    /// The chunks over granules `[from, to)`, wired.
    fn chunks(&self, from: usize, to: usize) -> Result<WiredAllocation, ExecutionError> {
        let buffers = (from..to)
            .map(|index| self.buffer(index, index + 1))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(WiredAllocation::wire(&self.residency, buffers))
    }

    /// The view over the first `granules` granules in `form`, wired, its
    /// table entries unwritten.
    ///
    /// # Panics
    /// Where `form` is whole over granules that may not be
    /// ([`Self::whole_at`]): the holder chooses the whole form only where it
    /// may be.
    fn buffers(&self, granules: usize, form: ViewForm) -> Result<GranuleView, ExecutionError> {
        Ok(match form {
            ViewForm::Whole => {
                assert!(self.whole_at(granules), "the whole form spans every granule, within the device's longest buffer");
                GranuleView::Whole(WiredAllocation::wire(&self.residency, vec![self.buffer(0, granules)?]))
            }
            ViewForm::Chunks => GranuleView::Chunks(self.chunks(0, granules)?),
        })
    }

    /// Write the table entries of the records in `buffers`: consecutive
    /// buffers of `span` bytes each, the first the range's `first`.
    fn write_entries(&self, first: usize, span: usize, buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>]) {
        let records = self.table.len() / RECORD_TABLE_ENTRY_BYTES;
        for (index, buffer) in (first as u64..).zip(buffers) {
            let (start, entries) = chunk_entries(index, span as u64, self.record as u64, records, buffer.gpuAddress());
            self.table.write_bytes(record_table_entry(start), &entries);
        }
    }

    /// Write the table entries of every record `view` spans.
    fn write_view_entries(&self, view: &GranuleView) {
        match view {
            GranuleView::Chunks(chunks) => self.write_entries(0, self.range.granule(), &chunks.buffers),
            GranuleView::Whole(whole) => {
                self.write_entries(0, self.range.granule() * self.range.granules(), &whole.buffers)
            }
        }
    }
}

impl Backing for Granules {
    type Error = ExecutionError;
    /// The buffers over the backed granules.
    type View = GranuleView;
    type Change = ChunkChange;
    /// The buffers a change no longer holds: a shrink's released chunks, or
    /// the form a change of form beside it replaced.
    type Retired = Option<GranuleView>;
    type Form = ViewForm;

    fn granule(&self) -> usize {
        self.range.granule()
    }

    fn granules(&self) -> usize {
        self.range.granules()
    }

    /// Map fresh zero granules `[from, to)`. On failure nothing new stays
    /// mapped.
    fn back(&self, from: usize, to: usize) -> Result<(), ExecutionError> {
        let range = &self.range;
        for index in from..to {
            if let Err(error) = unsafe { host::map_granule(range.at(index), range.granule(), range.page()) } {
                range.unback(from, index);
                return Err(error);
            }
        }
        Ok(())
    }

    fn unback(&self, from: usize, to: usize) {
        self.range.unback(from, to);
    }

    fn zero(&self, bytes: std::ops::Range<u64>) -> Result<(), ExecutionError> {
        self.range.zero(bytes);
        Ok(())
    }

    /// The buffers over the first `granules` granules in `form`, wired,
    /// their table entries written.
    fn view(&self, granules: usize, form: ViewForm) -> Result<GranuleView, ExecutionError> {
        let view = self.buffers(granules, form)?;
        self.write_view_entries(&view);
        Ok(view)
    }

    fn form(&self, view: &GranuleView) -> ViewForm {
        match view {
            GranuleView::Chunks(_) => ViewForm::Chunks,
            GranuleView::Whole(_) => ViewForm::Whole,
        }
    }

    /// From the whole form, a shrink's kept chunks are formed beside it, or
    /// left to the install where `forming` is in place; into it, its buffer
    /// is formed beside the chunks. Otherwise a growth of chunks forms its
    /// new chunks and writes their entries, and a shrink of chunks keeps the
    /// leading ones.
    ///
    /// # Panics
    /// Where `forming` is in place for anything but a shrink from the whole
    /// form: no other change can go on serving without its next buffers
    /// formed beside the view's.
    fn change(
        &self,
        (installed, from): (usize, ViewForm),
        granules: usize,
        form: ViewForm,
        forming: Forming,
    ) -> Result<ChunkChange, ExecutionError> {
        Ok(match (from, form, forming) {
            (ViewForm::Whole, ViewForm::Chunks, Forming::InPlace) => ChunkChange::InPlace(granules),
            (_, _, Forming::InPlace) => {
                panic!("only a shrink from the whole form forms its chunks in place, not {from:?} to {form:?}")
            }
            (ViewForm::Whole, _, _) | (_, ViewForm::Whole, _) => ChunkChange::Replace(self.buffers(granules, form)?),
            _ if granules > installed => {
                let chunks = self.chunks(installed, granules)?;
                self.write_entries(installed, self.range.granule(), &chunks.buffers);
                ChunkChange::Grow(chunks)
            }
            _ => ChunkChange::Shrink(granules),
        })
    }

    /// A change of form moves every record device work reaches through the
    /// table to the next form's buffers.
    fn rebinding(&self, change: &ChunkChange) -> Rebinding {
        match change {
            ChunkChange::Replace(_) => Rebinding::Beside,
            ChunkChange::InPlace(_) => Rebinding::InPlace,
            ChunkChange::Grow(_) | ChunkChange::Shrink(_) => Rebinding::None,
        }
    }

    /// A growth of chunks appends them; a shrink of chunks retires its tail
    /// chunks; a change of form beside writes its table entries and retires
    /// the form it replaces; one in place drops the whole form's buffer
    /// before it forms the kept chunks, so the two never count together.
    ///
    /// # Panics
    /// Where Metal refuses a buffer over granules the view already held in
    /// one buffer, past the device's limit on neither.
    fn apply(&self, view: &mut GranuleView, change: ChunkChange) -> Option<GranuleView> {
        fn chunks(view: &mut GranuleView) -> &mut WiredAllocation {
            match view {
                GranuleView::Chunks(chunks) => chunks,
                GranuleView::Whole(_) => unreachable!("a whole view changes only by a change of form"),
            }
        }
        match change {
            ChunkChange::Grow(grown) => {
                chunks(view).append(grown);
                None
            }
            ChunkChange::Shrink(granules) => Some(GranuleView::Chunks(chunks(view).split_off(granules))),
            ChunkChange::Replace(next) => {
                self.write_view_entries(&next);
                Some(std::mem::replace(view, next))
            }
            ChunkChange::InPlace(granules) => {
                assert!(matches!(view, GranuleView::Whole(_)), "a change of form in place is from the whole form");
                drop(std::mem::replace(view, GranuleView::Chunks(WiredAllocation::wire(&self.residency, Vec::new()))));
                let kept = self
                    .chunks(0, granules)
                    .unwrap_or_else(|error| panic!("Metal refused chunks over granules it held as one buffer: {error:?}"));
                *view = GranuleView::Chunks(kept);
                self.write_view_entries(view);
                None
            }
        }
    }
}

/// Mach virtual memory for reservations' granules.
mod host {
    use seismic_compiler::errors::ExecutionError;

    type KernReturn = libc::c_int;
    type MachPort = libc::c_uint;
    // <mach/memory_object_types.h>
    const MAP_MEM_NAMED_CREATE: libc::c_int = 0x020000;
    const VM_PROT_READ_WRITE: libc::c_int = 0x3;
    const VM_INHERIT_NONE: libc::c_uint = 2;
    // <mach/vm_statistics.h>
    const VM_FLAGS_FIXED: libc::c_int = 0x0000;
    const VM_FLAGS_OVERWRITE: libc::c_int = 0x4000;
    unsafe extern "C" {
        fn mach_make_memory_entry_64(
            target_task: MachPort,
            size: *mut u64,
            offset: u64,
            permission: libc::c_int,
            object_handle: *mut MachPort,
            parent_entry: MachPort,
        ) -> KernReturn;
        fn mach_vm_map(
            target_task: MachPort,
            address: *mut u64,
            size: u64,
            mask: u64,
            flags: libc::c_int,
            object: MachPort,
            offset: u64,
            copy: libc::c_int,
            cur_protection: libc::c_int,
            max_protection: libc::c_int,
            inheritance: libc::c_uint,
        ) -> KernReturn;
        fn mach_port_deallocate(task: MachPort, name: MachPort) -> KernReturn;
    }

    fn task() -> MachPort {
        #[allow(deprecated)]
        unsafe {
            libc::mach_task_self()
        }
    }

    /// Map one fresh, zero-filled granule of its own VM object at `address`,
    /// every `page`-byte host page of it backed: a committed granule is
    /// charged, so it is held as an allocated buffer's bytes are, and device
    /// work never takes a first-touch fault on it.
    ///
    /// # Safety
    /// The range lies inside a reservation of the caller's.
    pub(super) unsafe fn map_granule(address: *mut u8, bytes: usize, page: usize) -> Result<(), ExecutionError> {
        let failed = |what: &str, code: KernReturn| {
            ExecutionError::AllocationFailed(format!(
                "{what} for a reserved granule failed ({code})"
            ))
        };
        let mut size = bytes as u64;
        let mut entry: MachPort = 0;
        let code = unsafe {
            mach_make_memory_entry_64(
                task(),
                &mut size,
                0,
                MAP_MEM_NAMED_CREATE | VM_PROT_READ_WRITE,
                &mut entry,
                0,
            )
        };
        if code != 0 {
            return Err(failed("mach_make_memory_entry_64", code));
        }
        let mut at = address as u64;
        let code = unsafe {
            mach_vm_map(
                task(),
                &mut at,
                bytes as u64,
                0,
                VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                entry,
                0,
                0,
                VM_PROT_READ_WRITE,
                VM_PROT_READ_WRITE,
                VM_INHERIT_NONE,
            )
        };
        // The mapping holds the object; the entry's port is not needed.
        unsafe { mach_port_deallocate(task(), entry) };
        if code != 0 {
            return Err(failed("mach_vm_map", code));
        }
        // A write to each page makes the kernel back it with a zero page;
        // the granule's bytes stay zero.
        for offset in (0..bytes).step_by(page) {
            unsafe { std::ptr::write_volatile(address.add(offset), 0) };
        }
        Ok(())
    }
}

/// The device service: device plus one command queue.
#[derive(Clone)]
pub struct MetalDevice {
    handle: DeviceHandle,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Every buffer this device allocates. The queue carries the set, so
    /// command buffers inherit it without per-buffer encodings.
    residency: Arc<Residency>,
    /// A buffer every direct encoder declares it writes and no launch binds
    /// (`direct::DirectBatch`): hazard tracking then orders each encoder on
    /// the queue after the one before it.
    order: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    pub fn open(handle: DeviceHandle) -> Result<Self, TargetError> {
        crate::pooled(|| Self::form(handle))
    }

    fn form(handle: DeviceHandle) -> Result<Self, TargetError> {
        // Compilation is shared by every native entry formed on this device.
        // Metal defaults to a small compiler pool; allow the system to scale
        // compilation concurrency to the machine before forming any kernels.
        handle.raw().setShouldMaximizeConcurrentCompilation(true);
        let queue = handle.raw().newCommandQueue().ok_or_else(|| {
            TargetError::DeviceUnavailable("could not create a Metal command queue".into())
        })?;
        let residency = handle
            .raw()
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|error| TargetError::DeviceUnavailable(format!(
                "could not create a Metal residency set: {error}"
            )))?;
        queue.addResidencySet(&residency);
        // Metal rejects a zero-length buffer; nothing reads or writes it.
        let order = handle
            .raw()
            .newBufferWithLength_options(1, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| {
                TargetError::DeviceUnavailable("could not create the Metal order buffer".into())
            })?;
        Ok(Self {
            handle,
            queue,
            residency: Residency::new(residency),
            order,
        })
    }

    pub(crate) fn order(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.order
    }

    pub fn handle(&self) -> &DeviceHandle {
        &self.handle
    }

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Allocates a wired shared buffer of `bytes`. Metal rejects a
    /// zero-length buffer, so a zero-byte allocation reserves one byte while
    /// reporting its requested length.
    pub(crate) fn allocate_bytes(&self, bytes: u64) -> Result<WritableMetalBuffer, ExecutionError> {
        let length = usize::try_from(bytes.max(1)).map_err(|_| {
            ExecutionError::AllocationFailed(format!("{bytes} bytes exceed the host address space"))
        })?;
        let buffer = crate::pooled(|| {
            self.handle
                .raw()
                .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        })
        // Metal states no reason for a buffer it does not create; within
        // `maxBufferLength` it is the device's memory that refused it, a
        // capacity refusal as every backend reports one.
        .ok_or(ExecutionError::AllocationCapacity {
            required: bytes.into(),
            available: 0,
        })?;
        let wired = WiredAllocation::wire(&self.residency, vec![buffer.clone()]);
        Ok(WritableMetalBuffer(MetalBuffer {
            buffer,
            len: bytes,
            _host_owner: None,
            _wired: Some(Arc::new(wired)),
        }))
    }

    /// Zero-filled memory over a host address reservation of `reserved`
    /// bytes of `record`-byte records in `granule`-byte granules (chunks of
    /// whole records and whole [`RESERVATION_CHUNKS`] units, none smaller
    /// than [`RESERVATION_GRANULE`]), the leading
    /// `committed` bytes backed (in one buffer where they are every granule
    /// and it may be, [`ViewForm::Whole`]), and its resizer, which changes
    /// the backed prefix without moving its host address. `table` holds one device
    /// address per record of the range, written as chunks are formed. Each
    /// chunk is a buffer: `maxBufferLength` bounds a chunk (the runtime
    /// admits it), not the address range.
    pub fn allocate_reserved(
        &self,
        committed: u64,
        reserved: u64,
        granule: u64,
        record: u64,
    ) -> Result<(ReservedBuffer, MetalResizer), ExecutionError> {
        if granule < RESERVATION_GRANULE as u64
            || granule % CHUNK_UNIT != 0
            || record == 0
            || granule % record != 0
        {
            return Err(ExecutionError::AllocationFailed(format!(
                "a {granule}-byte granule of {record}-byte records is not whole {CHUNK_UNIT}-byte \
                 units and whole records, at least {RESERVATION_GRANULE} bytes"
            )));
        }
        // One device address a record of the range.
        let table = self.allocate_bytes(record_table_bytes(reserved / record).ok_or_else(|| {
            ExecutionError::AllocationFailed(format!("the record table of a {reserved}-byte reservation overflows"))
        })?)?;
        let backing = Granules {
            range: HostRange::reserve(reserved.max(committed), granule, PageSize::Host).map_err(|failure| {
                ExecutionError::AllocationFailed(format!("a Metal reservation's address range: {failure}"))
            })?,
            device: self.handle.clone(),
            residency: self.residency.clone(),
            table,
            record: record as usize,
        };
        let form = backing.formed_at(committed.div_ceil(granule) as usize);
        let (reservation, resizer) = Reserved::formed(backing, committed, form)?;
        Ok((ReservedBuffer { reservation }, resizer))
    }

    /// Wrap an immutable, page-aligned host mapping. The retained owner keeps
    /// the mapping live until the final Metal buffer reference is released,
    /// including references held by an in-flight command buffer.
    ///
    /// # Safety
    /// `pointer..pointer+length` must remain mapped and unchanged for the
    /// owner's lifetime. `pointer` and `length` must be host-page aligned.
    pub unsafe fn wrap_read_only_host_mapping(
        &self,
        pointer: NonNull<u8>,
        length: usize,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<MetalBuffer, ExecutionError> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = usize::try_from(page).map_err(|_| {
            ExecutionError::AllocationFailed("host page size is unavailable".into())
        })?;
        if length == 0 || pointer.as_ptr() as usize % page != 0 || length % page != 0 {
            return Err(ExecutionError::AllocationFailed(
                "Metal host mapping is not page-aligned".into(),
            ));
        }
        if length > self.handle.max_allocation_bytes() as usize {
            return Err(ExecutionError::AllocationFailed(
                "Metal host mapping exceeds maximum buffer length".into(),
            ));
        }
        let buffer = crate::pooled(|| unsafe {
            self.handle
                .raw()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer.cast(),
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        })
        .ok_or_else(|| {
            ExecutionError::AllocationFailed("Metal refused a read-only host mapping".into())
        })?;
        Ok(MetalBuffer {
            buffer,
            len: length as u64,
            _host_owner: Some(owner),
            _wired: None,
        })
    }
}

impl DeviceService<Metal> for MetalDevice {
    type Buffer = MetalBuffer;
    type Writable = WritableMetalBuffer;

    fn allocate(&self, bytes: u64, _alignment: u64) -> Result<Self::Writable, ExecutionError> {
        // The profile advertises the alignment returned by
        // `heapBufferSizeAndAlignWithLength:options:` for this same shared
        // buffer class; direct device buffers satisfy that requirement.
        self.allocate_bytes(bytes)
    }

    fn write(
        &self,
        buffer: &Self::Writable,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), ExecutionError> {
        buffer.write_bytes(offset, bytes);
        Ok(())
    }

    fn read(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError> {
        buffer.read_bytes(offset, into);
        Ok(())
    }

    fn buffer_len(&self, buffer: &Self::Buffer) -> u64 {
        buffer.len()
    }

    /// Every buffer: Metal storage is `MTLStorageModeShared`, whose
    /// `contents()` the host addresses for the buffer's whole lifetime.
    fn host_address(&self, buffer: &Self::Writable) -> Option<NonNull<u8>> {
        NonNull::new(buffer.host_base())
    }
}

#[cfg(test)]
mod mapped_tests {
    use super::*;

    struct Mapping {
        pointer: NonNull<u8>,
        len: usize,
    }
    unsafe impl Send for Mapping {}
    unsafe impl Sync for Mapping {}
    impl Drop for Mapping {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.pointer.as_ptr().cast(), self.len);
            }
        }
    }

    #[test]
    fn read_only_file_like_mapping_is_retained_by_metal_buffer() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle).unwrap();
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(pointer, libc::MAP_FAILED);
        let pointer = NonNull::new(pointer.cast::<u8>()).unwrap();
        unsafe {
            *pointer.as_ptr() = 73;
        }
        assert_eq!(
            unsafe { libc::mprotect(pointer.as_ptr().cast(), page, libc::PROT_READ) },
            0
        );
        let mapping = Arc::new(Mapping { pointer, len: page });
        let weak = Arc::downgrade(&mapping);
        let buffer =
            unsafe { device.wrap_read_only_host_mapping(pointer, page, mapping.clone()) }.unwrap();
        drop(mapping);
        assert!(weak.upgrade().is_some());
        assert_eq!(
            buffer.raw().contents().as_ptr() as usize,
            pointer.as_ptr() as usize
        );
        let mut byte = [0];
        buffer.read_bytes(0, &mut byte);
        assert_eq!(byte, [73]);
        // A host mapping stays file-backed: it never joins the residency set.
        assert_eq!(device.residency.lock().set.allocationCount(), 0);
        drop(buffer);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn an_allocated_buffer_is_wired_until_its_last_handle_is_dropped() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle).unwrap();
        let wired = |buffer: &MetalBuffer| {
            let set = &device.residency.lock().set;
            (
                set.containsAllocation(ProtocolObject::from_ref(&*buffer.raw())),
                set.allocationCount(),
                set.allocatedSize(),
            )
        };
        let first = device.allocate_bytes(1 << 20).unwrap();
        let second = device.allocate(1 << 16, 16).unwrap();
        assert_eq!(wired(&first), (true, 2, (1 << 20) + (1 << 16)));
        assert!(wired(&second).0);

        let handle = first.clone();
        drop(first);
        assert_eq!(wired(&handle), (true, 2, (1 << 20) + (1 << 16)));
        drop(handle);
        assert_eq!(wired(&second), (true, 1, 1 << 16));
        drop(second);
        assert_eq!(device.residency.lock().set.allocationCount(), 0);
    }

    #[test]
    fn residency_is_renewed_while_the_set_holds_an_allocation() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle).unwrap();
        let renewing = || {
            let wired = device.residency.lock();
            (wired.renewer.is_some(), wired.renewals)
        };
        assert_eq!(renewing(), (false, 0));

        let first = device.allocate_bytes(1 << 16).unwrap();
        let second = device.allocate_bytes(1 << 16).unwrap();
        std::thread::sleep(RESIDENCY_RENEWAL * 3 + RESIDENCY_RENEWAL / 2);
        let (running, renewals) = renewing();
        assert!(running);
        assert!((3..=5).contains(&renewals), "{renewals} renewals");

        drop(first);
        assert!(renewing().0);
        drop(second);
        let (running, stopped_at) = renewing();
        assert!(!running);
        std::thread::sleep(RESIDENCY_RENEWAL * 2);
        assert_eq!(renewing(), (false, stopped_at));

        let third = device.allocate_bytes(1 << 16).unwrap();
        std::thread::sleep(RESIDENCY_RENEWAL / 2);
        assert_eq!(renewing(), (true, stopped_at + 1));
        drop(third);
        assert!(!renewing().0);
    }

    #[test]
    fn stopping_the_renewer_does_not_wait_for_its_interval() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle).unwrap();
        for _ in 0..20 {
            let buffer = device.allocate_bytes(1 << 12).unwrap();
            let started = std::time::Instant::now();
            drop(buffer);
            assert!(started.elapsed() < RESIDENCY_RENEWAL / 2);
        }
        assert_eq!(device.residency.lock().generation, 20);
    }
}

#[cfg(test)]
mod reserved_tests {
    use super::*;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLLibrary, MTLSize,
    };

    const MIB: u64 = 1 << 20;

    /// The process's `phys_footprint`, the memory macOS charges the process
    /// with.
    fn footprint() -> u64 {
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage_info_v2>() };
        let status = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                (&raw mut usage).cast(),
            )
        };
        assert_eq!(status, 0);
        usage.ri_phys_footprint
    }

    /// Reads `bytes` of a buffer as `uint4` words on the GPU: their wrapping
    /// sum, and the GPU time of the read; through a record table, or bound
    /// directly.
    struct Reader {
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        direct: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        output: WritableMetalBuffer,
    }

    impl Reader {
        fn new(device: &MetalDevice) -> Self {
            let source = objc2_foundation::NSString::from_str(
                "#include <metal_stdlib>\nusing namespace metal;\n\
                 kernel void total(device const ulong* table [[buffer(0)]],\n\
                                   device atomic_uint* output [[buffer(1)]],\n\
                                   constant uint2& shape [[buffer(2)]],\n\
                                   uint index [[thread_position_in_grid]],\n\
                                   uint threads [[threads_per_grid]]) {\n\
                     uint4 sum = 0;\n\
                     for (uint word = index; word < shape.x; word += threads)\n\
                         sum += ((device const uint4*)table[word / shape.y])[word % shape.y];\n\
                     atomic_fetch_add_explicit(output, sum.x + sum.y + sum.z + sum.w, memory_order_relaxed);\n\
                 }\n\
                 kernel void direct(device const uint4* data [[buffer(0)]],\n\
                                    device atomic_uint* output [[buffer(1)]],\n\
                                    constant uint2& shape [[buffer(2)]],\n\
                                    uint index [[thread_position_in_grid]],\n\
                                    uint threads [[threads_per_grid]]) {\n\
                     uint4 sum = 0;\n\
                     for (uint word = index; word < shape.x; word += threads)\n\
                         sum += data[word];\n\
                     atomic_fetch_add_explicit(output, sum.x + sum.y + sum.z + sum.w, memory_order_relaxed);\n\
                 }",
            );
            let library = device
                .handle()
                .raw()
                .newLibraryWithSource_options_error(&source, None)
                .unwrap();
            let pipeline = |name: &str| {
                let function = library
                    .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                    .unwrap();
                device
                    .handle()
                    .raw()
                    .newComputePipelineStateWithFunction_error(&function)
                    .unwrap()
            };
            Self {
                pipeline: pipeline("total"),
                direct: pipeline("direct"),
                output: device.allocate_bytes(4).unwrap(),
            }
        }

        /// The `bytes` from `offset` of what `binding` binds, bound directly.
        fn bound_total(&self, device: &MetalDevice, binding: Binding<'_>, offset: u64, bytes: u64) -> u32 {
            self.read(device, &self.direct, binding, offset, [u32::try_from(bytes / 16).unwrap(), 0]).0
        }

        /// The first `bytes` of the `GRANULE`-byte records `table` addresses.
        fn total(&self, device: &MetalDevice, table: &MetalBuffer, bytes: u64) -> (u32, f64) {
            self.total_of(device, table, bytes, GRANULE)
        }

        /// The first `bytes` of the `record`-byte records `table` addresses.
        fn total_of(&self, device: &MetalDevice, table: &MetalBuffer, bytes: u64, record: u64) -> (u32, f64) {
            let shape = [u32::try_from(bytes / 16).unwrap(), (record / 16) as u32];
            self.read(device, &self.pipeline, Binding::Buffer(table), 0, shape)
        }

        /// Run `pipeline` over `binding` from `offset`, of `shape`.
        fn read(
            &self,
            device: &MetalDevice,
            pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
            binding: Binding<'_>,
            offset: u64,
            shape: [u32; 2],
        ) -> (u32, f64) {
            self.output.write_bytes(0, &[0; 4]);
            let command = device.queue().commandBuffer().unwrap();
            let encoder = command.computeCommandEncoder().unwrap();
            encoder.setComputePipelineState(pipeline);
            unsafe {
                binding.with(|buffer| encoder.setBuffer_offset_atIndex(Some(buffer), offset as usize, 0));
                encoder.setBuffer_offset_atIndex(Some(self.output.raw()), 0, 1);
                encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), 8, 2);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1 << 18,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let time = crate::timing::SubmissionTime::before_commit();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{:?}", command.error());
            let mut total = [0; 4];
            self.output.read_bytes(0, &mut total);
            let (start, end) = time.interval(command.GPUStartTime(), command.GPUEndTime()).unwrap();
            (u32::from_le_bytes(total), end - start)
        }
    }

    const GRANULE: u64 = RESERVATION_GRANULE as u64;

    /// A resize to `bytes` in `form`, a change of form's buffers formed
    /// where `forming` says, prepared.
    fn prepare_as(resizer: MetalResizer, bytes: u64, form: ViewForm, forming: Forming) -> seismic_reservation::Prepared<Granules> {
        resizer.prepare_as(bytes, form, forming).map_err(|(_, error)| error).unwrap()
    }

    /// One resize to `bytes` in `form`, formed beside: prepared, installed,
    /// released.
    fn resize_as(resizer: MetalResizer, bytes: u64, form: ViewForm) -> MetalResizer {
        prepare_as(resizer, bytes, form, Forming::Beside).install().release()
    }

    /// One resize of chunks to `bytes`.
    fn resize(resizer: MetalResizer, bytes: u64) -> MetalResizer {
        resize_as(resizer, bytes, ViewForm::Chunks)
    }

    fn pattern(bytes: u64) -> Vec<u8> {
        (0..bytes).map(|index| (index % 251) as u8 + 1).collect()
    }

    /// The wrapping sum of `bytes` as little-endian `u32` words.
    fn word_total(bytes: &[u8]) -> u32 {
        bytes.chunks_exact(4).fold(0u32, |sum, word| {
            sum.wrapping_add(u32::from_le_bytes(word.try_into().unwrap()))
        })
    }

    /// A reservation is one buffer over its range only while every granule
    /// is backed, in its chunks' place: formed at the reservation's forming
    /// with every granule, or by a growth to every granule in the whole
    /// form, and replaced by the kept chunks at a shrink's install, formed
    /// beside it before or in its place there. A growth to every granule in
    /// chunks stays chunks. Bound directly, from any offset, it reads what
    /// the host wrote, as the record table does in either form. The
    /// residency set holds each backed byte once.
    #[test]
    fn a_reservation_binds_as_one_buffer_while_every_granule_is_backed() {
        let device = crate::test_support::metal_device();
        let reader = Reader::new(&device);
        let whole = |buffer: &ReservedBuffer| matches!(buffer.reservation.state().view, GranuleView::Whole(_));
        // Bytes the residency set wires beyond what it wired with the
        // reservation empty, its table aside: the set does not count a
        // small table at its length (a 64-byte table read 64 bytes short,
        // M4 Pro).
        let wired_now = || device.residency.lock().set.allocatedSize() as u64;
        let reads = |buffer: &ReservedBuffer, contents: &[u8]| {
            let bytes = contents.len() as u64;
            assert_eq!(reader.bound_total(&device, Binding::Reserved(buffer), 0, bytes), word_total(contents));
            let (offset, tail) = (3 * GRANULE + 4096, &contents[(3 * GRANULE + 4096) as usize..]);
            assert_eq!(reader.bound_total(&device, Binding::Reserved(buffer), offset, tail.len() as u64), word_total(tail));
            assert_eq!(reader.total(&device, &buffer.table(), bytes).0, word_total(contents));
        };
        let (buffer, resizer) = device.allocate_reserved(0, 8 * GRANULE, GRANULE, GRANULE).unwrap();
        let others = wired_now();
        let wired = |_: &ReservedBuffer| wired_now() - others;
        let resizer = resize(resizer, 4 * GRANULE);
        assert!(!whole(&buffer), "a partly backed reservation has no one buffer");
        assert_eq!(wired(&buffer), 4 * GRANULE);
        let resizer = resize(resizer, 8 * GRANULE);
        assert!(!whole(&buffer), "a growth to every granule in chunks stays chunks");
        assert_eq!(wired(&buffer), 8 * GRANULE);
        assert_eq!(reader.total(&device, &buffer.table(), 8 * GRANULE).0, 0, "backed bytes read zero");
        let resizer = resize_as(resize(resizer, 7 * GRANULE), 8 * GRANULE, ViewForm::Whole);
        assert!(whole(&buffer), "the growth that backs the last granule in the whole form forms it");
        assert_eq!(wired(&buffer), 8 * GRANULE, "the one buffer takes the chunks' place");
        let contents = pattern(8 * GRANULE);
        buffer.write_bytes(0, &contents);
        reads(&buffer, &contents);
        let pending = prepare_as(resizer, 6 * GRANULE, ViewForm::Chunks, Forming::Beside).install();
        assert!(!whole(&buffer), "a shrink's install retires it, before its release");
        let resizer = pending.release();
        assert_eq!(wired(&buffer), 6 * GRANULE, "the kept chunks take its place");
        let kept = &contents[..(6 * GRANULE) as usize];
        assert_eq!(reader.total(&device, &buffer.table(), 6 * GRANULE).0, word_total(kept));
        let resizer = resize_as(resizer, 8 * GRANULE, ViewForm::Whole);
        assert!(whole(&buffer), "a growth back to every granule forms it again");
        assert_eq!(wired(&buffer), 8 * GRANULE);
        let contents: Vec<u8> = pattern(8 * GRANULE).into_iter().rev().collect();
        buffer.write_bytes(0, &contents);
        reads(&buffer, &contents);
        let prepared = prepare_as(resizer, 5 * GRANULE, ViewForm::Chunks, Forming::InPlace);
        assert_eq!(wired(&buffer), 8 * GRANULE, "a shrink in place forms nothing beside the one buffer");
        let pending = prepared.install();
        assert!(!whole(&buffer));
        assert_eq!(wired(&buffer), 5 * GRANULE, "its install forms the kept chunks in the one buffer's place");
        let resizer = pending.release();
        let kept = &contents[..(5 * GRANULE) as usize];
        assert_eq!(reader.total(&device, &buffer.table(), 5 * GRANULE).0, word_total(kept));
        let resizer = resize_as(resizer, 8 * GRANULE, ViewForm::Whole);
        assert!(whole(&buffer));
        drop((buffer, resizer));
        let (formed, resizer) = device.allocate_reserved(8 * GRANULE, 8 * GRANULE, GRANULE, GRANULE).unwrap();
        assert!(whole(&formed), "a reservation formed with every granule backed has it");
        drop((formed, resizer));
    }

    /// A reserved buffer resizes in place after GPU use: its host address
    /// and handles stay valid, kernels read the committed bytes through its
    /// record table correctly after a shrink and a grow, only committed
    /// chunks are wired, a shrink returns its pages to the system at once,
    /// and kept chunks keep their buffers.
    #[test]
    fn a_reserved_buffer_resizes_in_place_and_releases_what_it_shrinks() {
        let device = crate::test_support::metal_device();
        let reader = Reader::new(&device);
        // Chunk bytes the residency set wires beyond what was wired before,
        // the reservation's record table aside.
        let others = device.residency.lock().set.allocatedSize() as u64;
        let (buffer, resizer) = device
            .allocate_reserved(256 * MIB, 4096 * MIB, GRANULE, GRANULE)
            .unwrap();
        let table = buffer.table();
        let wired = || device.residency.lock().set.allocatedSize() as u64 - others - table.len();
        let handle = buffer.clone();
        let address = buffer.host_base();
        let first = buffer.gpu_address(0);
        assert_eq!(wired(), 256 * MIB, "only committed chunks are wired");
        let contents = pattern(256 * MIB);
        buffer.write_bytes(0, &contents);
        assert_eq!(
            reader.total(&device, &table, 256 * MIB).0,
            word_total(&contents)
        );

        let kept = 64 * MIB + 3;
        let before = footprint();
        let started = std::time::Instant::now();
        let resizer = resize(resizer, kept);
        let shrink = started.elapsed();
        let released = before.saturating_sub(footprint());
        eprintln!(
            "a 192 MiB shrink after GPU use took {shrink:?} and lowered phys_footprint by {released} bytes"
        );
        assert!(
            released.abs_diff(192 * MIB) <= 4 * MIB,
            "a 192 MiB shrink released {released} bytes"
        );
        assert_eq!(handle.len(), kept);
        assert_eq!(handle.host_base(), address);
        assert_eq!(wired(), 66 * MIB);
        assert_eq!(handle.gpu_address(0), first, "a kept chunk keeps its buffer");
        let mut read = vec![0; kept as usize];
        handle.read_bytes(0, &mut read);
        assert!(
            read == contents[..kept as usize],
            "kept bytes survive a shrink"
        );
        let words = 64 * MIB;
        assert_eq!(
            reader.total(&device, &table, words).0,
            word_total(&contents[..words as usize])
        );

        let mut grown = vec![1; (128 * MIB) as usize];
        let before = footprint();
        let resizer = resize(resizer, 128 * MIB);
        assert_eq!(buffer.host_base(), address);
        assert_eq!(wired(), 128 * MIB);
        buffer.read_bytes(0, &mut grown);
        let grown_by = footprint().saturating_sub(before);
        assert!(grown[..kept as usize] == contents[..kept as usize]);
        assert!(
            grown[kept as usize..].iter().all(|byte| *byte == 0),
            "bytes backed again read zero"
        );
        eprintln!("a 62 MiB growth raised phys_footprint by {grown_by} bytes once read");
        assert_eq!(
            reader.total(&device, &table, 128 * MIB).0,
            word_total(&grown)
        );
        drop((buffer, handle, resizer));
        assert_eq!(wired(), 0);
        drop(table);
    }

    /// What a reservation's chunk costs against its size
    /// ([`RESERVATION_GRANULE`]): for each chunk size, 1 GiB of chunk-sized
    /// records backed whole at once as a load forms it (`Reserved::new`),
    /// then GPU reads of it through its record table, and its unload; the
    /// load's parts by hand (mapping the granules, forming their buffers and
    /// table entries, the residency set's commit of them); and growths by
    /// one chunk through the resizer. Without the validation layer the
    /// other tests enable. Run on a quiet host:
    /// `cargo test -p seismic-metal --release --lib reservation_chunk_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement"]
    fn reservation_chunk_cost() {
        use std::time::Instant;
        const TOTAL: u64 = 1 << 30;
        const KIB: u64 = 1 << 10;
        let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
        let reader = Reader::new(&device);
        let ms = |started: Instant| started.elapsed().as_secs_f64() * 1e3;
        // Twice the load's bytes, so a load leaves a granule unbacked and is
        // formed as chunks.
        let granules = |chunk: u64| Granules {
            range: HostRange::reserve(2 * TOTAL, chunk, PageSize::Host).unwrap(),
            device: device.handle.clone(),
            residency: device.residency.clone(),
            table: device.allocate_bytes(record_table_bytes(2 * TOTAL / chunk).unwrap()).unwrap(),
            record: chunk as usize,
        };
        eprintln!(
            "{:>9} {:>7} | {:>9} {:>9} | {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} | {:>8} | {:>9}",
            "chunk", "chunks", "load ms", "us/chunk", "map ms", "buffers", "add ms", "remove", "read1 ms",
            "read2 ms", "unload", "grow us"
        );
        for chunk in [16 * KIB, 64 * KIB, 256 * KIB, MIB, 2 * MIB, 4 * MIB, 8 * MIB, 32 * MIB] {
            let chunks = (TOTAL / chunk) as usize;

            // A load: the whole commitment backed and formed at once.
            let started = Instant::now();
            let (reservation, resizer) = Reserved::new(granules(chunk), TOTAL).unwrap();
            let load = ms(started);
            assert!(matches!(&reservation.state().view, GranuleView::Chunks(formed) if formed.buffers.len() == chunks));
            let table = reservation.backing().table.clone();
            let (sum, first) = reader.total_of(&device, &table, TOTAL, chunk);
            assert_eq!(sum, 0, "backed bytes read zero");
            let (_, second) = reader.total_of(&device, &table, TOTAL, chunk);
            let started = Instant::now();
            drop((resizer, reservation));
            let unload = ms(started);
            drop(table);

            // The load's parts, over an empty reservation of the same range.
            let (reservation, resizer) = Reserved::new(granules(chunk), 0).unwrap();
            let backing = reservation.backing();
            let range = &backing.range;
            let started = Instant::now();
            backing.back(0, chunks).unwrap();
            let map = ms(started);
            let started = Instant::now();
            let wired = backing.chunks(0, chunks).unwrap();
            backing.write_entries(0, range.granule(), &wired.buffers);
            let form = ms(started);
            let buffers = wired.buffers.clone();
            let started = Instant::now();
            device.residency.remove(&buffers);
            let remove = ms(started);
            let started = Instant::now();
            device.residency.add(&buffers);
            let add = ms(started);
            drop((wired, buffers));
            range.unback(0, chunks);

            // Growths by one chunk, each prepared, installed and released.
            let mut resizer = resizer;
            let mut steps = Vec::new();
            for step in 1..=chunks.min(64) as u64 {
                let started = Instant::now();
                resizer = resize(resizer, step * chunk);
                steps.push(ms(started) * 1e3);
            }
            steps.sort_by(f64::total_cmp);
            drop((resize(resizer, 0), reservation));

            eprintln!(
                "{:>6} KiB {:>7} | {:>9.1} {:>9.2} | {:>8.1} {:>8.1} {:>8.1} {:>8.1} | {:>8.2} {:>8.2} | {:>8.1} | {:>9.1}",
                chunk / KIB,
                chunks,
                load,
                load * 1e3 / chunks as f64,
                map,
                form - add,
                add,
                remove,
                first * 1e3,
                second * 1e3,
                unload,
                steps[steps.len() / 2]
            );
        }
    }
}
