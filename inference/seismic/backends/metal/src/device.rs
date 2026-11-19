//! The Metal device service: one `MTLDevice`, one command queue, shared
//! storage-mode buffers (spec §12.3, R9).
//!
//! The service allocates, writes, reads and measures buffers. It infers
//! nothing about what a buffer holds.

use crate::Metal;
use objc2::rc::Retained;
use objc2::Message;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLCopyAllDevices, MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLDispatchType, MTLResidencySet, MTLResidencySetDescriptor, MTLResource,
    MTLResourceOptions, MTLResourceUsage,
};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::DeviceService;
use std::fmt;
use seismic_memory::entry::{Backed, DropScope, Entry, Store, Wiring};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
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
        seismic_lang::registry::test_device_reached("discovers the Metal devices");
        crate::pooled(|| {
            MTLCopyAllDevices()
                .into_iter()
                .map(|device| Self { device })
                .collect()
        })
    }

    /// The system default Metal device.
    pub fn system_default() -> Result<Self, TargetError> {
        seismic_lang::registry::test_device_reached("opens the default Metal device");
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

    /// Whether this device supports Metal residency sets (`MTLResidencySet`).
    /// Devices without them (e.g. Intel Mac GPUs) run with per-encoder
    /// resource declarations (`Keeping::Declared`).
    pub(crate) fn supports_residency_sets(&self) -> bool {
        crate::pooled(|| {
            self.device
                .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
                .is_ok()
        })
    }

    /// `MTLDevice.recommendedMaxWorkingSetSize`: Metal's advisory ceiling for
    /// this device's working set. Advice, not physical capacity.
    pub fn recommended_working_set_bytes(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// `MTLDevice.currentAllocatedSize`: bytes of resources this process has
    /// allocated on the device, including resources Seismic does not own.
    /// Each buffer counts its whole length; the device holds the working
    /// set to that count (its command buffers fail with out-of-memory
    /// beyond the recommended working set, M4 Pro, 2026-10-27).
    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    /// The device's refusal of `bytes` it did not allocate, its working set
    /// read at the refusal.
    pub(crate) fn refused(&self, bytes: u64) -> ExecutionError {
        ExecutionError::DeviceRefused {
            required: bytes,
            reading: seismic_compiler::errors::RefusalReading::WorkingSet {
                recommended: self.recommended_working_set_bytes(),
                allocated: self.current_allocated_bytes(),
            },
        }
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

/// What keeps every buffer the device allocates resident while the
/// process holds it: a residency set, which wires it, or, on a GPU that
/// cannot create one, the buffers themselves, declared to every compute
/// encoder ([`Keeping`]), which wires nothing ([`Residency::wiring`]).
///
/// A set's residency may lapse, macOS un-wiring its allocations until the
/// next request or GPU work wires them again; the GPU work wires the set
/// itself (the queue has it attached), so the set's residency is requested
/// only where a host reading needs it wired, and never on a timer:
/// - each change of the set (a buffer added or freed);
/// - a reading of the host's wired count
///   ([`MetalDevice::request_residency`], which the embedder's reading runs
///   before it reads the count).
///
/// A change holds the lock across its commit; a reading never waits on it:
/// where a change holds it, that change's own request, made after its
/// commit, wires the set again.
struct Residency {
    wired: Mutex<Wired>,
}

struct Wired {
    keeping: Keeping,
    /// The residency requests made: one per event that asks for one while
    /// the set holds anything.
    #[cfg(test)]
    requests: u64,
}

/// How the device's buffers are kept resident.
enum Keeping {
    /// A residency set the queue carries, so command buffers inherit it
    /// without per-encoder declarations.
    Set(Retained<ProtocolObject<dyn MTLResidencySet>>),
    /// A GPU that cannot create a residency set (an Intel Mac's
    /// `MTLIGAccelDevice`, Intel UHD Graphics 630 on macOS 15.8.2: "The
    /// operation couldn't be completed"): the buffers held, each declared
    /// to every compute encoder (`useResource:usage:`), as Metal kept
    /// resident what slab and record tables address before residency sets.
    /// A buffer is then backed by the first GPU work that declares it,
    /// not as it is allocated.
    Declared(Vec<Retained<ProtocolObject<dyn MTLBuffer>>>),
}

// The set and the buffers are used only under the residency mutex, on
// whichever thread holds it.
unsafe impl Send for Wired {}

impl Residency {
    fn new(keeping: Keeping) -> Arc<Self> {
        Arc::new(Self {
            wired: Mutex::new(Wired {
                keeping,
                #[cfg(test)]
                requests: 0,
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Wired> {
        self.wired.lock().expect("Metal residency lock poisoned")
    }

    /// How the OS holds a buffer kept resident here, which its charge
    /// states: wired by a set, pageable where declared, as the wire count
    /// of a GPU without a set (an Intel Mac's, which defines no wired
    /// limit) shows.
    fn wiring(&self) -> Wiring {
        match self.lock().keeping {
            Keeping::Set(_) => Wiring::Wired,
            Keeping::Declared(_) => Wiring::Pageable,
        }
    }

    /// Keep `buffer` resident from now. A set wires it now, on the calling
    /// thread: every page of it backed and resident before this returns. A
    /// buffer the set only holds is wired by the first GPU work that binds
    /// it, on the thread submitting that work (a 20 GiB buffer: +142 ms
    /// before its first read, M4 Pro, 2026-10-25), which an allocation is
    /// made away from.
    fn add(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let mut wired = self.lock();
        match &mut wired.keeping {
            Keeping::Set(set) => crate::pooled(|| set.addAllocation(ProtocolObject::from_ref(buffer))),
            Keeping::Declared(buffers) => return buffers.push(buffer.retain()),
        }
        wired.changed();
    }

    /// Keep `buffer` resident no longer.
    fn remove(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let mut wired = self.lock();
        match &mut wired.keeping {
            Keeping::Set(set) => crate::pooled(|| set.removeAllocation(ProtocolObject::from_ref(buffer))),
            Keeping::Declared(buffers) => return buffers.retain(|held| !std::ptr::eq(&**held, buffer)),
        }
        wired.changed();
    }

    /// A reading of the wired count: request the set's residency, without
    /// waiting. A change holding the lock requests it itself once its
    /// commit is made ([`Wired::changed`]), so that request stands for this
    /// one. Declared buffers have no residency to request between
    /// encoders.
    fn before_reading(&self) {
        match self.wired.try_lock() {
            Ok(mut wired) => wired.request(),
            Err(std::sync::TryLockError::WouldBlock) => {}
            Err(std::sync::TryLockError::Poisoned(_)) => panic!("Metal residency lock poisoned"),
        }
    }

    /// Make every buffer resident for `encoder`'s work: nothing for a set,
    /// which the queue carries; each declared buffer, read and written,
    /// otherwise.
    fn declare(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let wired = self.lock();
        if let Keeping::Declared(buffers) = &wired.keeping {
            for buffer in buffers {
                encoder.useResource_usage(ProtocolObject::from_ref(&**buffer), MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
        }
    }
}

impl Wired {
    /// The set changed: commit it, and request its residency again.
    fn changed(&mut self) {
        if let Keeping::Set(set) = &self.keeping {
            crate::pooled(|| set.commit());
        }
        self.request();
    }

    /// Request the set's residency where it holds anything.
    fn request(&mut self) {
        let Keeping::Set(set) = &self.keeping else { return };
        crate::pooled(|| {
            if set.allocationCount() > 0 {
                set.requestResidency();
                #[cfg(test)]
                {
                    self.requests += 1;
                }
            }
        });
    }

    /// How many buffers are kept resident.
    #[cfg(test)]
    fn count(&self) -> usize {
        match &self.keeping {
            Keeping::Set(set) => set.allocationCount() as usize,
            Keeping::Declared(buffers) => buffers.len(),
        }
    }
}

/// How a compute encoder is made ([`MetalDevice::compute_encoder`]).
pub(crate) enum EncoderKind<'a> {
    /// `computeCommandEncoder`: serial dispatch.
    Serial,
    /// `computeCommandEncoderWithDispatchType`.
    Dispatch(MTLDispatchType),
    /// `computeCommandEncoderWithDescriptor`: a pass, with its sample
    /// buffers.
    Pass(&'a MTLComputePassDescriptor),
}

/// The storage mode of `kind`, one of the backend's kinds
/// ([`crate::topology::kinds`]).
pub(crate) fn storage(kind: u32) -> MTLResourceOptions {
    match kind {
        crate::topology::SHARED_KIND => MTLResourceOptions::StorageModeShared,
        crate::topology::PRIVATE_KIND => MTLResourceOptions::StorageModePrivate,
        _ => panic!("invariant violated: Metal has no kind {kind}"),
    }
}

/// One Metal buffer with the bytes Metal took for it, charged exactly to an
/// entry ([`charged_buffer`]): its size class, admitted before the buffer
/// was made, and what Metal's own count of it took beyond that after. Freed
/// from the entry as it drops, with the buffer, in one change.
type ChargedBuffer = Backed<Retained<ProtocolObject<dyn MTLBuffer>>>;

/// A new buffer of `length` bytes in `options`' storage, its bytes charged
/// exactly to `entry` as `wiring` ([`ChargedBuffer`]): wired where it joins
/// the residency set. A refusal, the entry's or the device's, allocates
/// nothing and charges nothing.
fn charged_buffer(
    handle: &DeviceHandle,
    length: usize,
    options: MTLResourceOptions,
    entry: &Arc<dyn Entry>,
    wiring: Wiring,
) -> Result<ChargedBuffer, ExecutionError> {
    // What Metal takes for a buffer of `length`: its size class.
    let sized = handle.raw().heapBufferSizeAndAlignWithLength_options(length, options).size as u64;
    // The admission, the buffer and the record of what it took are one
    // change.
    Backed::take(entry, sized, wiring, |charging| {
        let buffer = crate::pooled(|| handle.raw().newBufferWithLength_options(length, options))
            // Metal states no reason for a buffer it does not create; within
            // `maxBufferLength` it is the device that refused it, with its
            // working set as it reads now.
            .ok_or_else(|| handle.refused(length as u64))?;
        // Metal's own count of the buffer, which its working set holds: what it
        // took beyond the size class, or the part it did not take.
        let allocated = buffer.allocatedSize() as u64;
        if allocated > sized {
            charging.took_more(allocated - sized);
        }
        if allocated < sized {
            charging.shrink(sized - allocated);
        }
        Ok::<_, ExecutionError>(buffer)
    })
}

/// A pool block's own anonymous mapping: its own VM object, so unmapping it
/// frees its pages at once, which a buffer Metal allocated does not (IOGPU
/// frees those at a time of its own, M4 Pro, macOS 26, 2026-11-14: a
/// touched 1 GiB buffer's footprint was held 3 s past its drop, and
/// `setPurgeableState(.empty)` or `madvise` discarded none of it).
struct Mapping {
    pointer: NonNull<u8>,
    len: usize,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    /// A zero mapping of `len` bytes, whole host pages.
    fn anonymous(len: usize) -> Result<Self, ExecutionError> {
        let pointer = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) };
        match (pointer != libc::MAP_FAILED).then(|| NonNull::new(pointer.cast::<u8>())).flatten() {
            Some(pointer) => Ok(Self { pointer, len }),
            None => Err(ExecutionError::AllocationFailed(format!("{len} bytes were not mapped: {}", std::io::Error::last_os_error()))),
        }
    }

    /// Every page of it backed, zero: written once each, on the calling
    /// thread, as a block's bytes are the process's the moment it returns.
    fn back(&self, page: usize) {
        for offset in (0..self.len).step_by(page) {
            // SAFETY: within the mapping this owns.
            unsafe { self.pointer.as_ptr().add(offset).write_volatile(0) };
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        let unmapped = unsafe { libc::munmap(self.pointer.as_ptr().cast(), self.len) };
        assert_eq!(unmapped, 0, "invariant violated: a mapping of {} bytes is unmapped: {}", self.len, std::io::Error::last_os_error());
    }
}

/// A new buffer of `length` bytes over a mapping of its own, rounded up to
/// whole host pages, every page backed, charged exactly to `entry` as
/// `wiring`: no-copy, so Metal takes exactly the mapping's bytes. A refusal allocates and
/// charges nothing.
fn mapped_buffer(handle: &DeviceHandle, length: usize, entry: &Arc<dyn Entry>, wiring: Wiring) -> Result<(ChargedBuffer, Mapping), ExecutionError> {
    let page = seismic_memory::host_page()
        .map_err(|error| ExecutionError::AllocationFailed(error.to_string()))?;
    let page = usize::try_from(page).map_err(|_| {
        ExecutionError::AllocationFailed("host page size exceeds the host address space".into())
    })?;
    let length = length.checked_next_multiple_of(page).ok_or_else(|| ExecutionError::AllocationFailed(format!("{length} bytes exceed the host address space")))?;
    let mut mapping = None;
    let buffer = Backed::take(entry, length as u64, wiring, |_| {
        let made = Mapping::anonymous(length)?;
        made.back(page);
        let buffer = crate::pooled(|| unsafe {
            handle.raw().newBufferWithBytesNoCopy_length_options_deallocator(made.pointer.cast(), length, MTLResourceOptions::StorageModeShared, None)
        })
        .ok_or_else(|| handle.refused(length as u64))?;
        mapping = Some(made);
        Ok::<_, ExecutionError>(buffer)
    })?;
    Ok((buffer, mapping.expect("a charged buffer has its mapping")))
}

/// A buffer's membership of its device's residency set, and the charge of
/// its bytes. It is requested resident from the commit that adds it until
/// this is dropped ([`Residency`]); its bytes are charged to the entry its
/// allocation named until then ([`MetalDevice::allocate_bytes`]).
pub struct WiredAllocation {
    residency: Arc<Residency>,
    buffer: ChargedBuffer,
    /// A pool block's mapping, unmapped as it is released ([`Self::drop`]):
    /// after the buffer over it. Held only for its drop.
    _mapping: Option<Mapping>,
    /// The change its drop runs in ([`DropScope`]): last, so the charge's
    /// free and the unmapping, an OS effect of the drop beside the
    /// buffer's release, are one change.
    scope: DropScope,
}

unsafe impl Send for WiredAllocation {}
unsafe impl Sync for WiredAllocation {}

impl WiredAllocation {
    fn wire(
        residency: &Arc<Residency>,
        buffer: ChargedBuffer,
        mapping: Option<Mapping>,
    ) -> Self {
        residency.add(&buffer);
        Self {
            residency: residency.clone(),
            buffer,
            _mapping: mapping,
            scope: DropScope::default(),
        }
    }
}

impl Drop for WiredAllocation {
    /// The buffer leaves the residency set first, then its fields drop in
    /// order: the buffer is released, and a block's mapping, which no work
    /// reads any more (the granule pool returns a block only after its
    /// granules' fences), is unmapped, its pages free within the drop. So
    /// our own release of a block shows in the next host reading with a
    /// release λ of 0, as anonymous memory's does.
    fn drop(&mut self) {
        self.scope.begin(self.buffer.entry());
        self.residency.remove(&self.buffer);
    }
}

/// One Metal buffer of fixed length: shared storage allocated by Metal
/// (wired) or wrapping a read-only host mapping, or private storage in the
/// device's own memory. Device work binds it ([`MetalBuffer::raw`]) and the
/// host reads it; the host writes only a [`WritableMetalBuffer`], the form
/// of a buffer Metal allocated.
#[derive(Clone)]
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: u64,
    storage: Storage,
    _host_owner: Option<Arc<dyn Any + Send + Sync>>,
    /// Absent for a read-only host mapping, whose pages stay file-backed
    /// and outside the residency set: resident only where it is bound, so
    /// no record table addresses it.
    _wired: Option<Arc<WiredAllocation>>,
}

/// How the host reaches a buffer's bytes.
#[derive(Clone)]
enum Storage {
    /// Shared storage: `contents()`, host-visible for the buffer's whole
    /// lifetime.
    Shared,
    /// Private storage, the device's own memory, which the host cannot map:
    /// through a blit of the device's queue, staged in shared storage
    /// ([`Staging`]).
    Private(Arc<Staging>),
}

/// The most bytes one staging copy holds in host RAM: its shared buffer is
/// charged to no ledger entry, so a host access of any size takes at most
/// this beside the bytes its caller holds.
const STAGING_CHUNK: usize = 16 << 20;

/// The device's single ordered command queue and the explicit
/// `maxCommandBufferCount` it was created with (`None` for Metal's default
/// `newCommandQueue()`).
struct CommandQueue {
    raw: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    capacity: Option<usize>,
}

// `MTLCommandQueue` is thread-safe; reads take a shared lock to create
// command buffers, and capacity growth takes an exclusive lock.
unsafe impl Send for CommandQueue {}
unsafe impl Sync for CommandQueue {}

/// The host's way to a private buffer's bytes: a blit on the device's
/// queue between it and a shared buffer of the bytes, completed before the
/// access returns, so it follows every command buffer committed before it
/// in the queue's order.
struct Staging {
    handle: DeviceHandle,
    queue: Arc<RwLock<CommandQueue>>,
}

// Metal's device and queue are thread-safe; a staging copy holds nothing
// across calls.
unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}

impl Staging {
    /// Encode `copy` in a blit of a new command buffer, commit it and wait
    /// for it.
    fn blit(&self, copy: impl FnOnce(&ProtocolObject<dyn MTLBlitCommandEncoder>)) -> Result<(), ExecutionError> {
        crate::pooled(|| {
            let command = {
                let queue = self.queue.read().expect("Metal command queue lock poisoned");
                queue.raw.commandBuffer()
            }
            .ok_or_else(|| ExecutionError::SubmissionFailed("Metal could not create a staging command buffer".into()))?;
            let blit = command
                .blitCommandEncoder()
                .ok_or_else(|| ExecutionError::SubmissionFailed("Metal could not create a staging blit encoder".into()))?;
            copy(&blit);
            blit.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            match command.error() {
                Some(error) => Err(ExecutionError::SynchronizationFailed(error.localizedDescription().to_string())),
                None => Ok(()),
            }
        })
    }

    /// A shared buffer of `length` bytes for a copy, `bytes` its contents
    /// where given.
    fn staged(&self, length: usize, bytes: Option<&[u8]>) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, ExecutionError> {
        crate::pooled(|| match bytes {
            Some(bytes) => unsafe {
                self.handle.raw().newBufferWithBytes_length_options(
                    NonNull::from(bytes).cast(),
                    length,
                    MTLResourceOptions::StorageModeShared,
                )
            },
            None => self.handle.raw().newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared),
        })
        .ok_or_else(|| self.handle.refused(length as u64))
    }

    /// Write `bytes` at `offset` of `buffer`, of `length` bytes, staged
    /// [`STAGING_CHUNK`] bytes at a time.
    fn write(&self, buffer: &ProtocolObject<dyn MTLBuffer>, length: u64, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        within(length, offset, bytes.len());
        for (index, chunk) in bytes.chunks(STAGING_CHUNK).enumerate() {
            let at = offset as usize + index * STAGING_CHUNK;
            let staged = self.staged(chunk.len(), Some(chunk))?;
            // SAFETY: both ranges lie in their buffers, which outlive the copy.
            self.blit(|blit| unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staged, 0, buffer, at, chunk.len())
            })?;
        }
        Ok(())
    }

    /// Read `into` from `offset` of `buffer`, of `length` bytes, staged
    /// [`STAGING_CHUNK`] bytes at a time.
    fn read(&self, buffer: &ProtocolObject<dyn MTLBuffer>, length: u64, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        within(length, offset, into.len());
        for (index, chunk) in into.chunks_mut(STAGING_CHUNK).enumerate() {
            let at = offset as usize + index * STAGING_CHUNK;
            let staged = self.staged(chunk.len(), None)?;
            // SAFETY: both ranges lie in their buffers, which outlive the copy.
            self.blit(|blit| unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(buffer, at, &staged, 0, chunk.len())
            })?;
            // SAFETY: the staged buffer is shared storage of `chunk.len()` bytes.
            unsafe { host_read(staged.contents().as_ptr().cast::<u8>(), chunk.len() as u64, 0, chunk) };
        }
        Ok(())
    }

    /// Zero every byte of `buffer`, of `length` bytes.
    fn zero(&self, buffer: &ProtocolObject<dyn MTLBuffer>, length: usize) -> Result<(), ExecutionError> {
        self.blit(|blit| blit.fillBuffer_range_value(buffer, NSRange::new(0, length), 0))
    }
}

/// A [`MetalBuffer`] Metal allocated, which the host also writes.
#[derive(Clone)]
pub struct WritableMetalBuffer(MetalBuffer);

impl WritableMetalBuffer {
    pub(crate) fn write_bytes(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        // Storage Metal allocated: host-writable for its lifetime.
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
    /// host-visible for the buffer's whole lifetime; none for private
    /// storage, which the host cannot map.
    fn host_base(&self) -> Option<NonNull<u8>> {
        match self.storage {
            Storage::Shared => NonNull::new(self.contents()),
            Storage::Private(_) => None,
        }
    }

    /// A shared buffer's `contents()`.
    fn contents(&self) -> *mut u8 {
        self.buffer.contents().as_ptr().cast::<u8>()
    }

    /// Write `bytes` at `offset` as device work does: the bound executor's
    /// own inputs, under the schedule's access plan; through a staging blit
    /// for private storage.
    ///
    /// # Safety
    /// The schedule plans these bytes as written by this work, so the
    /// buffer is one Metal allocated, never a read-only host mapping.
    pub(crate) unsafe fn write_device_bytes(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        match &self.storage {
            Storage::Private(staging) => staging.write(&self.buffer, self.len, offset, bytes),
            Storage::Shared => {
                unsafe { host_write(self.contents(), self.len, offset, bytes) };
                Ok(())
            }
        }
    }

    /// Read `into` from `offset`; through a staging blit for private
    /// storage.
    pub(crate) fn read_bytes(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        match &self.storage {
            Storage::Private(staging) => staging.read(&self.buffer, self.len, offset, into),
            Storage::Shared => {
                unsafe { host_read(self.contents(), self.len, offset, into) };
                Ok(())
            }
        }
    }
}

/// The host address of `[offset, offset + len)` of `length` bytes at
/// `base`. Precondition (FFI wrapper): the range lies inside them.
fn host_range(base: *mut u8, length: u64, offset: u64, len: usize) -> *mut u8 {
    within(length, offset, len);
    unsafe { base.add(offset as usize) }
}

/// Precondition of every host access: `[offset, offset + len)` lies in a
/// buffer of `length` bytes.
fn within(length: u64, offset: u64, len: usize) {
    let end = offset.checked_add(len as u64).filter(|end| *end <= length);
    assert!(
        end.is_some(),
        "Metal host access [{offset}, +{len}) exceeds the {length}-byte buffer"
    );
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

/// `handle`'s timestamp counter set, when it samples counters at stage
/// boundaries.
fn timestamp_counters(handle: &DeviceHandle) -> Option<Retained<ProtocolObject<dyn MTLCounterSet>>> {
    let raw = handle.raw();
    if !raw.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    // SAFETY: a static Metal string constant.
    let wanted = unsafe { MTLCommonCounterSetTimestamp };
    raw.counterSets()?.iter().find(|set| &*set.name() == wanted)
}

/// The device service: device plus one command queue.
#[derive(Clone)]
pub struct MetalDevice {
    handle: DeviceHandle,
    queue: Arc<RwLock<CommandQueue>>,
    /// What keeps every buffer this device allocates resident
    /// ([`Residency`]).
    residency: Arc<Residency>,
    /// A buffer every direct encoder declares it writes and no launch binds
    /// (`direct::DirectBatch`): hazard tracking then orders each encoder on
    /// the queue after the one before it.
    /// Its bytes are charged to the store's own entry, freed with it as the
    /// last clone of the device drops.
    order: Arc<ChargedBuffer>,
    /// The device's timestamp counter set, sampled at stage boundaries:
    /// every observed and timed batch, and every service probe, times its
    /// work by it (`direct::LaunchTimestamps`), so a device without it is
    /// refused at open.
    timestamps: Retained<ProtocolObject<dyn MTLCounterSet>>,
    /// The store the device's own memory lies in
    /// ([`crate::topology::own_kind`]): the embedder's ledger of it, whose own entry holds what the device
    /// allocates that no caller names.
    store: Arc<dyn Store>,
    /// The host's way to the device's private buffers ([`Storage::Private`]).
    staging: Arc<Staging>,
}

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    /// Device `handle`, its memory the store `store` ledgers: what it
    /// allocates of its own (the order buffer, its profile's probe) is
    /// charged to the store's own entry.
    pub fn open(handle: DeviceHandle, store: Arc<dyn Store>) -> Result<Self, TargetError> {
        crate::pooled(|| Self::form(handle, store))
    }

    fn form(handle: DeviceHandle, store: Arc<dyn Store>) -> Result<Self, TargetError> {
        // Compilation is shared by every native entry formed on this device.
        // Metal defaults to a small compiler pool; allow the system to scale
        // compilation concurrency to the machine before forming any kernels.
        handle.raw().setShouldMaximizeConcurrentCompilation(true);
        let timestamps = timestamp_counters(&handle).ok_or_else(|| {
            TargetError::DeviceUnavailable(format!(
                "Metal device `{}` cannot sample GPU timestamps at stage boundaries, which every timed \
                 and observed batch requires",
                handle.name()
            ))
        })?;
        let raw_queue = handle.raw().newCommandQueue().ok_or_else(|| {
            TargetError::DeviceUnavailable("could not create a Metal command queue".into())
        })?;
        // A GPU that cannot create a residency set runs all the same, its
        // buffers declared to each encoder instead ([`Keeping::Declared`]).
        let keeping = match handle.raw().newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()) {
            Ok(set) => {
                raw_queue.addResidencySet(&set);
                Keeping::Set(set)
            }
            Err(_) => Keeping::Declared(Vec::new()),
        };
        // Metal rejects a zero-length buffer; nothing reads or writes it. It
        // stays outside the residency set: pageable.
        let order =
            charged_buffer(&handle, 1, MTLResourceOptions::StorageModePrivate, &store.own(), Wiring::Pageable).map_err(|error| {
                TargetError::DeviceUnavailable(format!("could not create the Metal order buffer: {error}"))
            })?;
        let queue = Arc::new(RwLock::new(CommandQueue { raw: raw_queue, capacity: None }));
        let staging = Arc::new(Staging { handle: handle.clone(), queue: queue.clone() });
        Ok(Self {
            handle,
            queue,
            residency: Residency::new(keeping),
            order: Arc::new(order),
            timestamps,
            store,
            staging,
        })
    }

    /// Ensure the device's command queue can hold at least `count` uncompleted
    /// command buffers at once (`newCommandQueueWithMaxCommandBufferCount:`).
    ///
    /// Metal's default `newCommandQueue()` caps uncompleted command buffers,
    /// after which `commandBuffer()` blocks until an earlier buffer completes
    /// (deadlocking behind an unopened host gate). Called at load time while
    /// the queue is idle: when no explicit capacity is set yet or the current
    /// capacity is below `count`, drains the current queue, opens a
    /// replacement sized to `count`, and reattaches the residency set so every
    /// already-allocated buffer remains resident.
    pub fn ensure_command_buffer_capacity(&self, count: usize) -> Result<(), ExecutionError> {
        if count == 0 {
            return Ok(());
        }
        if self
            .queue
            .read()
            .expect("Metal command queue lock poisoned")
            .capacity
            .is_some_and(|current| current >= count)
        {
            return Ok(());
        }
        let mut queue = self
            .queue
            .write()
            .expect("Metal command queue lock poisoned");
        if queue.capacity.is_some_and(|current| current >= count) {
            return Ok(());
        }
        crate::pooled(|| {
            let replacement = self
                .handle
                .raw()
                .newCommandQueueWithMaxCommandBufferCount(count)
                .ok_or_else(|| {
                    ExecutionError::AllocationFailed(format!(
                        "Metal could not create a command queue for {count} command buffers"
                    ))
                })?;
            if let Some(drain) = queue.raw.commandBuffer() {
                drain.commit();
                drain.waitUntilCompleted();
            }
            let wired = self.residency.lock();
            if let Keeping::Set(set) = &wired.keeping {
                replacement.addResidencySet(set);
            }
            queue.raw = replacement;
            queue.capacity = Some(count);
            Ok(())
        })
    }

    /// The explicit `maxCommandBufferCount` the queue was sized to, or `None`
    /// while using Metal's default `newCommandQueue()`.
    pub fn command_buffer_capacity(&self) -> Option<usize> {
        self.queue
            .read()
            .expect("Metal command queue lock poisoned")
            .capacity
    }

    /// The device's timestamp counter set, sampled at stage boundaries.
    pub(crate) fn timestamp_counters(&self) -> &Retained<ProtocolObject<dyn MTLCounterSet>> {
        &self.timestamps
    }

    pub(crate) fn order(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.order
    }

    pub fn handle(&self) -> &DeviceHandle {
        &self.handle
    }

    /// The store the device was opened with.
    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    /// A new command buffer on the device's queue: the one way GPU work
    /// begins on the device. None where Metal creates none.
    pub(crate) fn command_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        crate::pooled(|| {
            self.queue
                .read()
                .expect("Metal command queue lock poisoned")
                .raw
                .commandBuffer()
        })
    }

    /// A new compute encoder of `command`, of `kind`, with every buffer the
    /// device keeps resident made resident for its work ([`Residency`]):
    /// the one way compute work is encoded on the device. None where Metal
    /// creates none.
    pub(crate) fn compute_encoder(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        kind: EncoderKind<'_>,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let encoder = match kind {
            EncoderKind::Serial => command.computeCommandEncoder(),
            EncoderKind::Dispatch(dispatch) => command.computeCommandEncoderWithDispatchType(dispatch),
            EncoderKind::Pass(pass) => command.computeCommandEncoderWithDescriptor(pass),
        }?;
        self.residency.declare(&encoder);
        Some(encoder)
    }

    /// Request the residency set resident: run before each reading of the
    /// host's wired count ([`Residency`]), without waiting on a change of
    /// the set (which requests it itself). The request re-wires a lapsed set
    /// by a mechanism that reports no completion, so the wired count read
    /// right after may still fall short of our wired bytes
    /// (`residency_probe` measures it).
    pub fn request_residency(&self) {
        self.residency.before_reading();
    }

    /// Allocates a wired shared buffer of `bytes`, every page backed and
    /// zero (Metal initializes a new buffer's contents to zero, and wiring
    /// it backs every page before it returns), its bytes charged exactly to
    /// `entry` until its last handle drops. Metal rejects a zero-length
    /// buffer, so a zero-byte allocation takes one byte's buffer while
    /// reporting its requested length.
    pub(crate) fn allocate_bytes(
        &self,
        bytes: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<WritableMetalBuffer, ExecutionError> {
        self.allocate_shared(bytes, entry, false)
    }

    /// [`Self::allocate_bytes`], a pool block's where `block`: over a
    /// mapping of its own, unmapped as it is released ([`Mapping`]).
    fn allocate_shared(
        &self,
        bytes: u64,
        entry: &Arc<dyn Entry>,
        block: bool,
    ) -> Result<WritableMetalBuffer, ExecutionError> {
        let length = usize::try_from(bytes.max(1)).map_err(|_| {
            ExecutionError::AllocationFailed(format!("{bytes} bytes exceed the host address space"))
        })?;
        // Kept resident before it is handed out, wired where a set keeps it.
        let wiring = self.residency.wiring();
        let (charged, mapping) = match block {
            true => {
                let (charged, mapping) = mapped_buffer(&self.handle, length, entry, wiring)?;
                (charged, Some(mapping))
            }
            false => (charged_buffer(&self.handle, length, MTLResourceOptions::StorageModeShared, entry, wiring)?, None),
        };
        Ok(self.held(charged, mapping, bytes, Storage::Shared))
    }

    /// A private buffer of `bytes` in the device's own memory, its bytes as
    /// `fill` says (Metal leaves a private buffer's contents undefined, so a
    /// blit zeroes it for [`seismic_memory::SlabFill::Zeroed`]; skipped for
    /// [`seismic_memory::SlabFill::Unwritten`] so allocation never
    /// synchronously waits on the device command queue), its bytes charged
    /// exactly to `entry` as device memory the driver holds
    /// ([`Wiring::Pageable`]), kept resident for the device's work as every
    /// buffer is ([`Residency`]).
    fn allocate_private(&self, bytes: u64, entry: &Arc<dyn Entry>, fill: seismic_memory::SlabFill) -> Result<WritableMetalBuffer, ExecutionError> {
        let length = usize::try_from(bytes.max(1)).map_err(|_| {
            ExecutionError::AllocationFailed(format!("{bytes} bytes exceed the host address space"))
        })?;
        let charged = charged_buffer(&self.handle, length, storage(crate::topology::PRIVATE_KIND), entry, Wiring::Pageable)?;
        if fill == seismic_memory::SlabFill::Zeroed {
            self.staging.zero(&charged, length)?;
        }
        Ok(self.held(charged, None, bytes, Storage::Private(self.staging.clone())))
    }

    /// `charged`, of `len` requested bytes, kept resident until its last
    /// handle drops ([`WiredAllocation`]).
    fn held(&self, charged: ChargedBuffer, mapping: Option<Mapping>, len: u64, storage: Storage) -> WritableMetalBuffer {
        let buffer = Retained::clone(&charged);
        let wired = WiredAllocation::wire(&self.residency, charged, mapping);
        WritableMetalBuffer(MetalBuffer {
            buffer,
            len,
            storage,
            _host_owner: None,
            _wired: Some(Arc::new(wired)),
        })
    }

    /// A buffer of `bytes` of `kind`, one of the device's kinds
    /// ([`crate::topology::kinds`]): private storage, or shared storage, a
    /// pool block's over a mapping of its own where `block`
    /// ([`Self::allocate_shared`]).
    fn allocate_kind(&self, kind: u32, bytes: u64, entry: &Arc<dyn Entry>, block: bool, fill: seismic_memory::SlabFill) -> Result<WritableMetalBuffer, ExecutionError> {
        let unified = self.handle.has_unified_memory();
        assert!(
            crate::topology::kinds(unified).iter().any(|own| own.id == kind),
            "invariant violated: a Metal allocation of kind {kind}, which the device (unified memory: {unified}) does not have"
        );
        match kind {
            crate::topology::PRIVATE_KIND => self.allocate_private(bytes, entry, fill),
            _ => self.allocate_shared(bytes, entry, block),
        }
    }

    /// A buffer of `bytes` of the device's own memory, the kind its store
    /// ledgers ([`crate::topology::own_kind`]), zero, charged exactly to
    /// `entry`.
    pub(crate) fn allocate_own(&self, bytes: u64, entry: &Arc<dyn Entry>) -> Result<WritableMetalBuffer, ExecutionError> {
        self.allocate_kind(crate::topology::own_kind(self.handle.has_unified_memory()), bytes, entry, false, seismic_memory::SlabFill::Zeroed)
    }

    /// One block of a granule pool (`seismic_memory::pool::Blocks`) of
    /// `kind`, its bytes as `fill` says, charged exactly to `entry`: shared
    /// storage over a mapping of its own ([`Self::allocate_shared`]), its pages
    /// returned to the OS within the drop of its last handle
    /// ([`WiredAllocation`]'s drop), or private storage
    /// ([`Self::allocate_private`]).
    pub fn allocate_block(
        &self,
        kind: u32,
        bytes: u64,
        entry: &Arc<dyn Entry>,
        fill: seismic_memory::SlabFill,
    ) -> Result<WritableMetalBuffer, ExecutionError> {
        self.allocate_kind(kind, bytes, entry, true, fill)
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
        let page = seismic_memory::host_page()
            .map_err(|error| ExecutionError::AllocationFailed(error.to_string()))?;
        let page = usize::try_from(page).map_err(|_| {
            ExecutionError::AllocationFailed("host page size exceeds the host address space".into())
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
            storage: Storage::Shared,
            _host_owner: Some(owner),
            _wired: None,
        })
    }
}

impl DeviceService<Metal> for MetalDevice {
    type Buffer = MetalBuffer;
    type Writable = WritableMetalBuffer;

    /// Shared storage, or private storage on a device with memory of its
    /// own ([`crate::topology::kinds`]).
    fn allocate(
        &self,
        kind: u32,
        bytes: u64,
        _alignment: u64,
        entry: &Arc<dyn Entry>,
    ) -> Result<Self::Writable, ExecutionError> {
        // The profile advertises the alignment returned by
        // `heapBufferSizeAndAlignWithLength:options:` for this same shared
        // buffer class; direct device buffers satisfy that requirement.
        self.allocate_kind(kind, bytes, entry, false, seismic_memory::SlabFill::Zeroed)
    }

    fn write(
        &self,
        buffer: &Self::Writable,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), ExecutionError> {
        buffer.write_bytes(offset, bytes)
    }

    fn read(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError> {
        buffer.read_bytes(offset, into)
    }

    fn buffer_len(&self, buffer: &Self::Buffer) -> u64 {
        buffer.len()
    }

    /// A shared buffer's `contents()`, which the host addresses for the
    /// buffer's whole lifetime; none for a private one.
    fn host_address(&self, buffer: &Self::Writable) -> Option<NonNull<u8>> {
        buffer.host_base()
    }
}

#[cfg(test)]
mod mapped_tests {
    use super::*;
    use std::time::{Duration, Instant};

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
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
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
        buffer.read_bytes(0, &mut byte).unwrap();
        assert_eq!(byte, [73]);
        // A host mapping stays file-backed: it never joins the residency set.
        assert_eq!(device.residency.lock().count(), 0);
        drop(buffer);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn an_allocated_buffer_is_wired_until_its_last_handle_is_dropped() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let wired = |buffer: &MetalBuffer| {
            let wired = device.residency.lock();
            let Keeping::Set(set) = &wired.keeping else { panic!("this GPU keeps a residency set") };
            (
                set.containsAllocation(ProtocolObject::from_ref(&*buffer.raw())),
                set.allocationCount(),
                set.allocatedSize(),
            )
        };
        let first = device.allocate_bytes(1 << 20, &crate::test_support::entry()).unwrap();
        let second = device.allocate(crate::topology::SHARED_KIND, 1 << 16, 16, &crate::test_support::entry()).unwrap();
        assert_eq!(wired(&first), (true, 2, (1 << 20) + (1 << 16)));
        assert!(wired(&second).0);

        let handle = first.clone();
        drop(first);
        assert_eq!(wired(&handle), (true, 2, (1 << 20) + (1 << 16)));
        drop(handle);
        assert_eq!(wired(&second), (true, 1, 1 << 16));
        drop(second);
        assert_eq!(device.residency.lock().count(), 0);
    }

    /// A private buffer, as a device with memory of its own places device
    /// work's buffers in (any Metal device allocates one): zero, kept
    /// resident, unmapped by the host, its bytes written and read through
    /// staging blits.
    #[test]
    fn a_private_buffer_is_zero_resident_and_reached_through_staging() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let buffer = device.allocate_private(3 << 12, &crate::test_support::entry(), seismic_memory::SlabFill::Zeroed).unwrap();
        assert!(<MetalDevice as DeviceService<Metal>>::host_address(&device, &buffer).is_none());
        assert_eq!(device.residency.lock().count(), 1);
        let mut read = vec![1u8; 3 << 12];
        buffer.read_bytes(0, &mut read).unwrap();
        assert!(read.iter().all(|byte| *byte == 0));
        let written = (0..5000u32).map(|index| index as u8).collect::<Vec<_>>();
        buffer.write_bytes(4097, &written).unwrap();
        let mut back = vec![0u8; written.len() + 2];
        buffer.read_bytes(4096, &mut back).unwrap();
        assert_eq!((back[0], &back[1..=written.len()], back[written.len() + 1]), (0, &written[..], 0));
        drop(buffer);
        assert_eq!(device.residency.lock().count(), 0);
        // An access longer than one staging copy is staged a chunk at a
        // time, each landing at its own offset.
        let long = device.allocate_private((STAGING_CHUNK + (3 << 12)) as u64, &crate::test_support::entry(), seismic_memory::SlabFill::Zeroed).unwrap();
        let written = (0..STAGING_CHUNK + 4096).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        long.write_bytes(4097, &written).unwrap();
        let mut back = vec![0u8; written.len()];
        long.read_bytes(4097, &mut back).unwrap();
        assert!(back == written, "a chunked access reads back what it wrote");
    }

    /// A buffer's bytes are charged exactly as Metal counts them (its size
    /// class: a one-byte buffer takes a page) until its last handle drops,
    /// and a refused charge allocates nothing.
    #[test]
    fn a_buffer_is_charged_what_metal_takes_for_it_until_its_last_handle_drops() {
        use seismic_memory::entry::Tally;
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let tally = Tally::unbounded(seismic_topology::StoreIdentity::HostRam);
        let entry: Arc<dyn Entry> = Arc::new(tally.clone());
        for bytes in [1u64, 1 << 16, (1 << 20) + 3] {
            let buffer = device.allocate_bytes(bytes, &entry).unwrap();
            assert_eq!(tally.held(), buffer.raw().allocatedSize() as u64);
            assert!(tally.held() >= bytes);
            let held = buffer.clone();
            drop(buffer);
            assert!(tally.held() >= bytes, "a handle keeps the charge");
            drop(held);
            assert_eq!(tally.held(), 0);
        }
        tally.bound(0);
        let before = device.handle().current_allocated_bytes();
        assert!(matches!(
            device.allocate_block(crate::topology::SHARED_KIND, 1 << 20, &entry, seismic_memory::SlabFill::Zeroed),
            Err(ExecutionError::AllocationCapacity { .. })
        ));
        assert_eq!(device.handle().current_allocated_bytes(), before);
    }

    /// The set's residency is requested on each change while it holds
    /// anything, and on a reading of the wired count; nothing requests
    /// between our own events, however long the device idles, and a command
    /// buffer requests nothing.
    #[test]
    fn residency_is_requested_on_changes_and_readings() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let requests = || device.residency.lock().requests;
        device.request_residency();
        drop(device.command_buffer().unwrap());
        assert_eq!(requests(), 0, "an empty set requests nothing");
        let first = device.allocate_bytes(1 << 16, &crate::test_support::entry()).unwrap();
        let second = device.allocate_bytes(1 << 16, &crate::test_support::entry()).unwrap();
        assert_eq!(requests(), 2, "one request per buffer added");
        let command = device.command_buffer().unwrap();
        assert_eq!(requests(), 2, "a command buffer requests nothing");
        crate::pooled(|| {
            command.commit();
            command.waitUntilCompleted();
        });
        device.request_residency();
        assert_eq!(requests(), 3, "a reading requests it");
        drop(first);
        assert_eq!(requests(), 4, "a free leaving the set holding a buffer requests it again");
        drop(second);
        assert_eq!(requests(), 4, "an empty set requests nothing");
        assert_eq!(device.residency.lock().count(), 0);
    }

    /// A reading of the wired count while a change of the set holds its lock
    /// (a resize mid-commit, stood in for by the lock held here) neither
    /// waits for it nor requests: the change requests after its commit.
    #[test]
    fn a_reading_never_waits_behind_a_change_of_the_set() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = Arc::new(MetalDevice::open(handle, crate::test_support::store()).unwrap());
        let _buffer = device.allocate_bytes(1 << 16, &crate::test_support::entry()).unwrap();
        let changing = device.residency.lock();
        let requests = changing.requests;
        let (began, begun) = std::sync::mpsc::channel();
        let reader = {
            let device = device.clone();
            std::thread::spawn(move || {
                device.request_residency();
                began.send(()).unwrap();
            })
        };
        let returned = begun.recv_timeout(Duration::from_secs(10));
        drop(changing);
        reader.join().unwrap();
        assert!(returned.is_ok(), "a reading waited on the set's lock");
        assert_eq!(device.residency.lock().requests, requests, "the change in progress requests for it");
    }

    /// The probe's set. It calls the set lapsed where the machine's wired
    /// count has fallen by at least half of it since the set was held.
    const PROBE_BYTES: u64 = 256 << 20;
    const PROBE_POLL: Duration = Duration::from_millis(50);
    const PROBE_SPAN: Duration = Duration::from_secs(5);

    /// An empty command buffer's submit, wait and the time it takes.
    fn submit_empty(device: &MetalDevice) -> Duration {
        let started = Instant::now();
        let command = device.command_buffer().unwrap();
        crate::pooled(|| {
            command.commit();
            command.waitUntilCompleted();
        });
        started.elapsed()
    }

    /// Idle until the live wired count falls `PROBE_BYTES / 2` below `held`
    /// (the set's observed lapse) or `PROBE_SPAN` passes: the time to the
    /// fall, or none where the OS never un-wired within the span.
    fn idle_until_lapsed(held: u64) -> Option<Duration> {
        let started = Instant::now();
        while started.elapsed() < PROBE_SPAN {
            std::thread::sleep(PROBE_POLL);
            if wire_count() + PROBE_BYTES / 2 <= held {
                return Some(started.elapsed());
            }
        }
        None
    }

    /// Measures what the residency requests cost and show on this host,
    /// asserting none of it: the set's observed lapse while idle, the time
    /// of one request and the live wired count right after it, and the first
    /// submit after an idle span with and without a request before it. Each
    /// figure is one `RESIDENCY_PROBE key=value` line.
    #[test]
    fn residency_probe() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let buffer = device.allocate_bytes(PROBE_BYTES, &crate::test_support::entry()).unwrap();
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let ones = vec![1u8; page as usize];
        for offset in (0..PROBE_BYTES).step_by(page as usize) {
            buffer.write_bytes(offset, &ones).unwrap();
        }
        submit_empty(&device);
        let held = wire_count();
        assert!(held > 0, "the live wired count reads zero");
        let ms = |time: Duration| time.as_secs_f64() * 1e3;
        eprintln!("RESIDENCY_PROBE bytes={PROBE_BYTES} source={:?} held_wired={held}", wire_source());

        let lapse = idle_until_lapsed(held);
        eprintln!("RESIDENCY_PROBE observed_lapse_ms={:?}", lapse.map(ms));
        let idle = wire_count();
        let started = Instant::now();
        device.request_residency();
        let took = started.elapsed();
        let right_after = wire_count();
        eprintln!(
            "RESIDENCY_PROBE idle_wired={idle} request_ms={:.3} wired_right_after={right_after} wired_at_once={}",
            ms(took),
            right_after + PROBE_BYTES / 2 > held,
        );

        // The first submit after an idle span, the span ended by the fall of
        // the wired count (or its bound); the set is wired again by the
        // submit that ends the first span, before the second begins.
        submit_empty(&device);
        idle_until_lapsed(wire_count());
        eprintln!("RESIDENCY_PROBE first_submit_after_idle_ms={:.3}", ms(submit_empty(&device)));
        idle_until_lapsed(wire_count());
        device.request_residency();
        eprintln!("RESIDENCY_PROBE first_submit_after_idle_with_request_ms={:.3}", ms(submit_empty(&device)));
        drop(buffer);
    }

    /// The machine's wired bytes now: `vm.page_wired_count` where macOS
    /// defines it (26 on), read live; before, `host_statistics64`'s
    /// `wire_count`, which may serve a snapshot taken up to 1 s before.
    fn wire_count() -> u64 {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        match live_wire_count() {
            Some(pages) => pages * page,
            None => {
                let mut statistics: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
                let mut count = libc::HOST_VM_INFO64_COUNT;
                #[allow(deprecated)]
                let host = unsafe { libc::mach_host_self() };
                let status = unsafe {
                    libc::host_statistics64(host, libc::HOST_VM_INFO64, (&mut statistics as *mut libc::vm_statistics64).cast(), &mut count)
                };
                assert_eq!(status, libc::KERN_SUCCESS, "host_statistics64 returned {status}");
                u64::from(statistics.wire_count) * page
            }
        }
    }

    fn wire_source() -> &'static str {
        match live_wire_count() {
            Some(_) => "vm.page_wired_count (live)",
            None => "host_statistics64 (a snapshot up to 1 s old)",
        }
    }

    fn live_wire_count() -> Option<u64> {
        let mut bytes = [0u8; 8];
        let mut length = bytes.len();
        let status = unsafe {
            libc::sysctlbyname(c"vm.page_wired_count".as_ptr(), bytes.as_mut_ptr().cast(), &mut length, std::ptr::null_mut(), 0)
        };
        match (status, length) {
            (0, 4) => Some(u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))),
            (0, 8) => Some(u64::from_ne_bytes(bytes)),
            (0, length) => panic!("vm.page_wired_count returned {length} bytes"),
            _ => None,
        }
    }

    /// A buffer in the residency set is charged wired; the order buffer,
    /// outside it, pageable.
    #[test]
    fn buffers_in_the_residency_set_are_charged_wired() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let entry = crate::test_support::entry();
        let buffer = device.allocate_bytes(1 << 16, &entry).unwrap();
        let wired = buffer._wired.as_ref().expect("an allocated buffer is in the set");
        assert_eq!(wired.buffer.charge().wiring(), Wiring::Wired);
        assert_eq!(device.order.charge().wiring(), Wiring::Pageable);
    }

    /// A pool's block is one buffer in the residency set: with a block of
    /// one granule (Metal bounds no allocation count, so `least` is none),
    /// the set holds one buffer per granule the pool holds. The count, and
    /// what a change of the set costs at it, is what a load's pool puts
    /// there.
    #[test]
    fn a_pool_block_is_one_buffer_in_the_residency_set() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let block = 1u64 << 21;
        let count = || device.residency.lock().count() as u64;
        let before = count();
        let started = std::time::Instant::now();
        let blocks = (0..512)
            .map(|_| device.allocate_block(crate::topology::SHARED_KIND, block, &crate::test_support::entry(), seismic_memory::SlabFill::Zeroed).unwrap())
            .collect::<Vec<_>>();
        let added = started.elapsed();
        assert_eq!(count() - before, blocks.len() as u64, "one buffer per block");
        let started = std::time::Instant::now();
        drop(blocks);
        let freed = started.elapsed();
        assert_eq!(count(), before);
        eprintln!(
            "the residency set at 512 one-granule blocks of {block} bytes: a block's add {:.3} ms, its free {:.3} ms",
            added.as_secs_f64() * 1e3 / 512.0,
            freed.as_secs_f64() * 1e3 / 512.0
        );
    }

    /// A pool block's pages are returned inside its release: its last
    /// handle's drop unmaps the block's own mapping, so this process's
    /// physical footprint (`task_vm_info.phys_footprint`, read live and
    /// counting the IOGPU buffers it holds) falls by at least the block
    /// between a reading just before the drop and one just after.
    #[test]
    fn a_released_block_leaves_the_footprint_within_the_release() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let block = 1u64 << 30;
        let buffer = device.allocate_block(crate::topology::SHARED_KIND, block, &crate::test_support::entry(), seismic_memory::SlabFill::Zeroed).unwrap();
        // Touched: every page the process's.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let ones = vec![1u8; page as usize];
        for offset in (0..block).step_by(page as usize) {
            buffer.write_bytes(offset, &ones).unwrap();
        }
        let before = footprint();
        drop(buffer);
        let after = footprint();
        assert!(
            before >= after + block,
            "the release took {} of the block's {block} bytes off the footprint",
            before.saturating_sub(after)
        );
    }

    /// `task_vm_info` (`TASK_VM_INFO`) through `phys_footprint`, laid out as
    /// the kernel writes it.
    #[allow(dead_code)]
    #[repr(C)]
    #[derive(Default)]
    struct TaskVm {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }

    /// This process's physical footprint now, read live.
    fn footprint() -> u64 {
        const TASK_VM_INFO: libc::task_flavor_t = 22;
        let mut vm = TaskVm::default();
        let words = (std::mem::size_of::<TaskVm>() / std::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
        let mut count = words;
        #[allow(deprecated)]
        let task = unsafe { libc::mach_task_self() };
        let status = unsafe { libc::task_info(task, TASK_VM_INFO, (&mut vm as *mut TaskVm).cast(), &mut count) };
        assert_eq!(status, libc::KERN_SUCCESS, "task_info TASK_VM_INFO returned {status}");
        assert_eq!(count, words, "task_vm_info reaches phys_footprint");
        vm.phys_footprint
    }
}
