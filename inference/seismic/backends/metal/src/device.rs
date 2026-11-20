//! The Metal device service: one `MTLDevice`, one command queue, shared
//! storage-mode buffers (spec §12.3, R9).
//!
//! The service allocates, writes, reads and measures buffers. It infers
//! nothing about what a buffer holds.

use crate::Metal;
use objc2::rc::Retained;
use objc2::Message;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSRange};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferError,
    MTLCommandBufferErrorDomain, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLCopyAllDevices, MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLDispatchType, MTLResidencySet, MTLResidencySetDescriptor, MTLResource,
    MTLResourceOptions, MTLResourceUsage,
};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::DeviceService;
use std::fmt;
use seismic_memory::entry::{Backed, DropInChange, Entry, ScopedDrop, Store, Wiring};
use std::sync::{Arc, Mutex, RwLock};
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
    /// resource declarations (`Residency::Declared`).
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

    /// Map a failed command buffer's `NSError`: Metal reports working-set
    /// exhaustion as `MTLCommandBufferErrorOutOfMemory`, which is a capacity
    /// refusal; any other command-buffer error goes through `fallback`.
    pub(crate) fn command_buffer_error(
        &self,
        error: &NSError,
        fallback: fn(String) -> ExecutionError,
    ) -> ExecutionError {
        classify_command_buffer_error(error, || self.refused(0), fallback)
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
/// encoder, which wires nothing ([`Residency::wiring`]).
///
/// The mode is fixed once at [`MetalDevice::open`] and lives outside the
/// mutex so hot-path checks ([`Residency::declare`] on a set,
/// [`Residency::before_reading`] on declared buffers, [`Residency::wiring`])
/// take no lock.
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
/// A change holds the set lock across its commit; a reading never waits on it:
/// where a change holds it, that change's own request, made after its
/// commit, wires the set again.
enum Residency {
    /// A residency set the queue carries, so command buffers inherit it
    /// without per-encoder declarations.
    Set(Mutex<SetState>),
    /// A GPU that cannot create a residency set (an Intel Mac's
    /// `MTLIGAccelDevice`, Intel UHD Graphics 630 on macOS 15.8.2: "The
    /// operation couldn't be completed"): the buffers held, each declared
    /// to every compute encoder (`useResource:usage:`), as Metal kept
    /// resident what slab and record tables address before residency sets.
    /// A buffer is then backed by the first GPU work that declares it,
    /// not as it is allocated.
    Declared(Mutex<DeclaredBuffers>),
}

struct SetState {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// The residency requests made: one per event that asks for one while
    /// the set holds anything.
    #[cfg(test)]
    requests: u64,
}

// The residency set is used only under its mutex, on whichever thread holds it.
unsafe impl Send for SetState {}

struct DeclaredBuffers(Vec<Retained<ProtocolObject<dyn MTLBuffer>>>);

// The declared buffer list is used only under its mutex, on whichever thread holds it.
unsafe impl Send for DeclaredBuffers {}

impl Residency {
    fn set(set: Retained<ProtocolObject<dyn MTLResidencySet>>) -> Arc<Self> {
        Arc::new(Self::Set(Mutex::new(SetState {
            set,
            #[cfg(test)]
            requests: 0,
        })))
    }

    fn declared() -> Arc<Self> {
        Arc::new(Self::Declared(Mutex::new(DeclaredBuffers(Vec::new()))))
    }

    /// How the OS holds a buffer kept resident here, which its charge
    /// states: wired by a set, pageable where declared, as the wire count
    /// of a GPU without a set (an Intel Mac's, which defines no wired
    /// limit) shows.
    fn wiring(&self) -> Wiring {
        match self {
            Self::Set(_) => Wiring::Wired,
            Self::Declared(_) => Wiring::Pageable,
        }
    }

    /// Attach the residency set to `queue`, where a set is used.
    fn attach(&self, queue: &ProtocolObject<dyn MTLCommandQueue>) {
        if let Self::Set(state) = self {
            let state = state.lock().expect("Metal residency lock poisoned");
            queue.addResidencySet(&state.set);
        }
    }

    /// Keep `buffer` resident from now. A set wires it now, on the calling
    /// thread: every page of it backed and resident before this returns. A
    /// buffer the set only holds is wired by the first GPU work that binds
    /// it, on the thread submitting that work (a 20 GiB buffer: +142 ms
    /// before its first read, M4 Pro, 2026-10-25), which an allocation is
    /// made away from.
    fn add(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        match self {
            Self::Set(state) => {
                let mut state = state.lock().expect("Metal residency lock poisoned");
                crate::pooled(|| state.set.addAllocation(ProtocolObject::from_ref(buffer)));
                state.changed();
            }
            Self::Declared(buffers) => {
                buffers
                    .lock()
                    .expect("Metal residency lock poisoned")
                    .0
                    .push(buffer.retain());
            }
        }
    }

    /// Keep `buffer` resident no longer.
    fn remove(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        match self {
            Self::Set(state) => {
                let mut state = state.lock().expect("Metal residency lock poisoned");
                crate::pooled(|| state.set.removeAllocation(ProtocolObject::from_ref(buffer)));
                state.changed();
            }
            Self::Declared(buffers) => {
                buffers
                    .lock()
                    .expect("Metal residency lock poisoned")
                    .0
                    .retain(|held| !std::ptr::eq(&**held, buffer));
            }
        }
    }

    /// A reading of the wired count: request the set's residency, without
    /// waiting. A change holding the lock requests it itself once its
    /// commit is made ([`SetState::changed`]), so that request stands for
    /// this one. Declared buffers have no residency to request between
    /// encoders.
    fn before_reading(&self) {
        let Self::Set(state) = self else { return };
        match state.try_lock() {
            Ok(mut state) => state.request(),
            Err(std::sync::TryLockError::WouldBlock) => {}
            Err(std::sync::TryLockError::Poisoned(_)) => panic!("Metal residency lock poisoned"),
        }
    }

    /// Make every buffer resident for `encoder`'s work: nothing for a set,
    /// which the queue carries; each declared buffer, read and written,
    /// otherwise.
    fn declare(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let Self::Declared(buffers) = self else { return };
        let buffers = buffers.lock().expect("Metal residency lock poisoned");
        for buffer in &buffers.0 {
            encoder.useResource_usage(
                ProtocolObject::from_ref(&**buffer),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
        }
    }

    /// How many buffers are kept resident.
    #[cfg(test)]
    fn count(&self) -> usize {
        match self {
            Self::Set(state) => state
                .lock()
                .expect("Metal residency lock poisoned")
                .set
                .allocationCount() as usize,
            Self::Declared(buffers) => buffers
                .lock()
                .expect("Metal residency lock poisoned")
                .0
                .len(),
        }
    }

    #[cfg(test)]
    fn lock_set(&self) -> std::sync::MutexGuard<'_, SetState> {
        let Self::Set(state) = self else {
            panic!("this GPU keeps a residency set");
        };
        state.lock().expect("Metal residency lock poisoned")
    }

    #[cfg(test)]
    fn requests(&self) -> u64 {
        match self {
            Self::Set(state) => state.lock().expect("Metal residency lock poisoned").requests,
            Self::Declared(_) => 0,
        }
    }
}

impl SetState {
    /// The set changed: commit it, and request its residency again.
    fn changed(&mut self) {
        crate::pooled(|| self.set.commit());
        self.request();
    }

    /// Request the set's residency where it holds anything.
    fn request(&mut self) {
        crate::pooled(|| {
            if self.set.allocationCount() > 0 {
                self.set.requestResidency();
                #[cfg(test)]
                {
                    self.requests += 1;
                }
            }
        });
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

fn classify_command_buffer_error(
    error: &NSError,
    refused: impl FnOnce() -> ExecutionError,
    fallback: fn(String) -> ExecutionError,
) -> ExecutionError {
    let domain = unsafe { MTLCommandBufferErrorDomain };
    if *error.domain() == *domain && error.code() == MTLCommandBufferError::OutOfMemory.0 as isize {
        refused()
    } else {
        fallback(error.localizedDescription().to_string())
    }
}

fn classify_mmap_error(
    len: usize,
    error: std::io::Error,
    refused: impl FnOnce(u64) -> ExecutionError,
) -> ExecutionError {
    if error.raw_os_error() == Some(libc::ENOMEM) {
        refused(len as u64)
    } else {
        ExecutionError::AllocationFailed(format!("{len} bytes were not mapped: {error}"))
    }
}

impl Mapping {
    /// A zero mapping of `len` bytes, whole host pages.
    fn anonymous(handle: &DeviceHandle, len: usize) -> Result<Self, ExecutionError> {
        let pointer = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) };
        match (pointer != libc::MAP_FAILED).then(|| NonNull::new(pointer.cast::<u8>())).flatten() {
            Some(pointer) => Ok(Self { pointer, len }),
            None => Err(classify_mmap_error(len, std::io::Error::last_os_error(), |bytes| handle.refused(bytes))),
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

/// The host page size as `usize`.
fn host_page_usize() -> Result<usize, ExecutionError> {
    let page = seismic_memory::host_page()
        .map_err(|error| ExecutionError::AllocationFailed(error.to_string()))?;
    usize::try_from(page).map_err(|_| {
        ExecutionError::AllocationFailed("host page size exceeds the host address space".into())
    })
}

/// A new buffer of `length` bytes over a mapping of its own, rounded up to
/// whole host pages (`page`), every page backed, charged exactly to `entry` as
/// `wiring`: no-copy, so Metal takes exactly the mapping's bytes. A refusal allocates and
/// charges nothing.
fn mapped_buffer(
    handle: &DeviceHandle,
    length: usize,
    page: usize,
    entry: &Arc<dyn Entry>,
    wiring: Wiring,
) -> Result<(ChargedBuffer, Mapping), ExecutionError> {
    let length = length.checked_next_multiple_of(page).ok_or_else(|| ExecutionError::AllocationFailed(format!("{length} bytes exceed the host address space")))?;
    let mut mapping = None;
    let buffer = Backed::take(entry, length as u64, wiring, |_| {
        let made = Mapping::anonymous(handle, length)?;
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

/// A charged `StorageModeShared` buffer of `length` bytes: backed by its own
/// anonymous [`Mapping`] whenever `force_mapped` or `length > page` so its
/// pages return to the OS within the drop (`release lag == 0`), avoiding IOGPU's
/// ~3 s release lag on `newBufferWithLength_options`. Sub-page buffers
/// (`length <= page` and `!force_mapped`) stay on `charged_buffer` to avoid
/// per-buffer `mmap` VMAs.
fn shared_buffer(
    handle: &DeviceHandle,
    length: usize,
    entry: &Arc<dyn Entry>,
    wiring: Wiring,
    force_mapped: bool,
) -> Result<(ChargedBuffer, Option<Mapping>), ExecutionError> {
    let page = host_page_usize()?;
    if force_mapped || length > page {
        let (charged, mapping) = mapped_buffer(handle, length, page, entry, wiring)?;
        Ok((charged, Some(mapping)))
    } else {
        Ok((charged_buffer(handle, length, MTLResourceOptions::StorageModeShared, entry, wiring)?, None))
    }
}

struct WiredAllocationInner {
    residency: Arc<Residency>,
    buffer: ChargedBuffer,
    /// A pool block's mapping, unmapped as it is released:
    /// after the buffer over it. Held only for its drop.
    _mapping: Option<Mapping>,
}

impl DropInChange for WiredAllocationInner {
    fn drop_entry(&self) -> Option<&Arc<dyn Entry>> {
        Some(self.buffer.entry())
    }

    fn before_drop(&mut self) {
        self.residency.remove(&self.buffer);
    }
}

/// A buffer's membership of its device's residency set, and the charge of
/// its bytes. It is requested resident from the commit that adds it until
/// this is dropped ([`Residency`]); its bytes are charged to the entry its
/// allocation named until then ([`MetalDevice::allocate_bytes`]).
pub struct WiredAllocation(ScopedDrop<WiredAllocationInner>);

unsafe impl Send for WiredAllocation {}
unsafe impl Sync for WiredAllocation {}

impl WiredAllocation {
    fn wire(
        residency: &Arc<Residency>,
        buffer: ChargedBuffer,
        mapping: Option<Mapping>,
    ) -> Self {
        residency.add(&buffer);
        Self(ScopedDrop::new(WiredAllocationInner {
            residency: residency.clone(),
            buffer,
            _mapping: mapping,
        }))
    }

    #[cfg(test)]
    fn buffer(&self) -> &ChargedBuffer {
        &self.0.buffer
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

/// The most bytes one staging copy holds in host RAM: each chunk is charged
/// to the device store's own ledger entry (`Wiring::Pageable`) and freed as
/// soon as its blit finishes, so a host access of any size takes at most this
/// beside the bytes its caller holds.
const STAGING_CHUNK: usize = 16 << 20;

struct StagedBufferInner {
    buffer: ChargedBuffer,
    _mapping: Option<Mapping>,
}

impl DropInChange for StagedBufferInner {
    fn drop_entry(&self) -> Option<&Arc<dyn Entry>> {
        Some(self.buffer.entry())
    }
}

/// One transient staging buffer held for the duration of a single blit chunk.
/// Its charge release and optional mapping unmap run together inside one change.
struct StagedBuffer(ScopedDrop<StagedBufferInner>);

impl std::ops::Deref for StagedBuffer {
    type Target = StagedBufferInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
/// queue between it and a charged shared buffer of the bytes, completed before
/// the access returns, so it follows every command buffer committed before it
/// in the queue's order.
struct Staging {
    handle: DeviceHandle,
    queue: Arc<RwLock<CommandQueue>>,
    entry: Arc<dyn Entry>,
}

// Metal's device, queue, and ledger entry are thread-safe; a staging copy
// holds nothing across calls.
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
                Some(error) => Err(self.handle.command_buffer_error(&error, ExecutionError::SynchronizationFailed)),
                None => Ok(()),
            }
        })
    }

    /// A charged shared buffer of `length` bytes for a copy, `bytes` its
    /// contents where given. Uses [`shared_buffer`] so chunks above page size
    /// are unmapped immediately on drop (`release lag == 0`).
    fn staged(&self, length: usize, bytes: Option<&[u8]>) -> Result<StagedBuffer, ExecutionError> {
        let (buffer, mapping) = shared_buffer(&self.handle, length, &self.entry, Wiring::Pageable, false)?;
        if let Some(bytes) = bytes {
            // SAFETY: `buffer` is a writable shared buffer of at least `length` bytes.
            unsafe { host_write(buffer.contents().as_ptr().cast::<u8>(), length as u64, 0, bytes) };
        }
        Ok(StagedBuffer(ScopedDrop::new(StagedBufferInner {
            buffer,
            _mapping: mapping,
        })))
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
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staged.buffer, 0, buffer, at, chunk.len())
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
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(buffer, at, &staged.buffer, 0, chunk.len())
            })?;
            // SAFETY: the staged buffer is shared storage of `chunk.len()` bytes.
            unsafe { host_read(staged.buffer.contents().as_ptr().cast::<u8>(), chunk.len() as u64, 0, chunk) };
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
        // buffers declared to each encoder instead ([`Residency::Declared`]).
        let residency = match handle.raw().newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new()) {
            Ok(set) => {
                raw_queue.addResidencySet(&set);
                Residency::set(set)
            }
            Err(_) => Residency::declared(),
        };
        // Metal rejects a zero-length buffer; nothing reads or writes it. It
        // stays outside the residency set: pageable.
        let order =
            charged_buffer(&handle, 1, MTLResourceOptions::StorageModePrivate, &store.own(), Wiring::Pageable).map_err(|error| {
                TargetError::DeviceUnavailable(format!("could not create the Metal order buffer: {error}"))
            })?;
        let queue = Arc::new(RwLock::new(CommandQueue { raw: raw_queue, capacity: None }));
        let staging = Arc::new(Staging {
            handle: handle.clone(),
            queue: queue.clone(),
            entry: store.own(),
        });
        Ok(Self {
            handle,
            queue,
            residency,
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
            self.residency.attach(&replacement);
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

    /// [`Self::allocate_bytes`], routed through [`shared_buffer`]: backed by
    /// its own [`Mapping`] whenever `block` or larger than a host page so its
    /// pages return to the OS within the drop (`release lag == 0`).
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
        // The wired charge and the residency commit in held() must become
        // observable together. Backed::take joins this change.
        let _change = entry.change();
        let (charged, mapping) = shared_buffer(&self.handle, length, entry, wiring, block)?;
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
        assert_eq!(device.residency.count(), 0);
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
            let state = device.residency.lock_set();
            (
                state.set.containsAllocation(ProtocolObject::from_ref(&*buffer.raw())),
                state.set.allocationCount(),
                state.set.allocatedSize(),
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
        assert_eq!(device.residency.count(), 0);
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
        assert_eq!(device.residency.count(), 1);
        let mut read = vec![1u8; 3 << 12];
        buffer.read_bytes(0, &mut read).unwrap();
        assert!(read.iter().all(|byte| *byte == 0));
        let written = (0..5000u32).map(|index| index as u8).collect::<Vec<_>>();
        buffer.write_bytes(4097, &written).unwrap();
        let mut back = vec![0u8; written.len() + 2];
        buffer.read_bytes(4096, &mut back).unwrap();
        assert_eq!((back[0], &back[1..=written.len()], back[written.len() + 1]), (0, &written[..], 0));
        drop(buffer);
        assert_eq!(device.residency.count(), 0);
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

    /// Observe every boundary at which a host reading could see the ledger.
    #[test]
    fn a_shared_buffers_charge_and_residency_change_together() {
        use seismic_memory::entry::{Change, Refused, Tally};
        use seismic_topology::StoreIdentity;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone)]
        struct Observed {
            tally: Tally,
            residency: Arc<Residency>,
            depth: Arc<AtomicUsize>,
        }

        struct Boundary(Observed);

        impl Drop for Boundary {
            fn drop(&mut self) {
                let entry = &self.0;
                if entry.depth.fetch_sub(1, Ordering::Relaxed) == 1 {
                    let resident = entry.residency.count() > 0;
                    assert_eq!(entry.tally.held() > 0, resident);
                }
            }
        }

        impl Entry for Observed {
            fn store(&self) -> &StoreIdentity { self.tally.store() }
            fn change(&self) -> Change {
                self.depth.fetch_add(1, Ordering::Relaxed);
                Change::held(Boundary(self.clone()))
            }
            fn on_close(&self, closed: Box<dyn FnOnce() + Send>) { self.tally.on_close(closed) }
            fn closed(&self) -> bool { self.tally.closed() }
            fn take(&self, change: &Change, bytes: u64, wiring: Wiring) -> Result<(), Refused> {
                self.tally.take(change, bytes, wiring)
            }
            fn took(&self, change: &Change, bytes: u64, wiring: Wiring) { self.tally.took(change, bytes, wiring) }
            fn freed(&self, change: &Change, bytes: u64, wiring: Wiring) { self.tally.freed(change, bytes, wiring) }
        }

        let device = crate::test_support::metal_device();
        let record = seismic_lang::registry::TestDevice::opened(seismic_lang::registry::BackendName::Metal);
        let entry: Arc<dyn Entry> = Arc::new(Observed {
            tally: Tally::unbounded(StoreIdentity::HostRam),
            residency: device.residency.clone(),
            depth: Arc::new(AtomicUsize::new(0)),
        });
        for block in [false, true] {
            let buffer = device.allocate_shared(1 << 16, &entry, block).unwrap();
            assert_eq!(device.residency.count(), 1);
            drop(buffer);
        }
        if let Some(record) = record {
            record.ran();
        }
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
        let requests = || device.residency.requests();
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
        assert_eq!(device.residency.count(), 0);
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
        let changing = device.residency.lock_set();
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
        assert_eq!(device.residency.requests(), requests, "the change in progress requests for it");
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
        assert_eq!(wired.buffer().charge().wiring(), Wiring::Wired);
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
        let count = || device.residency.count() as u64;
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

    /// A non-pool shared buffer above page size is also backed by its own
    /// mapping (`shared_buffer`), so dropping its last handle returns its
    /// pages to the OS within the drop (`release lag == 0`) rather than
    /// suffering IOGPU's ~3 s release lag.
    #[test]
    fn a_released_non_pool_buffer_leaves_the_footprint_within_the_release() {
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let device = MetalDevice::open(handle, crate::test_support::store()).unwrap();
        let bytes = 256u64 << 20;
        let buffer = device.allocate_bytes(bytes, &crate::test_support::entry()).unwrap();
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let ones = vec![1u8; page as usize];
        for offset in (0..bytes).step_by(page as usize) {
            buffer.write_bytes(offset, &ones).unwrap();
        }
        let before = footprint();
        drop(buffer);
        let after = footprint();
        assert!(
            before >= after + bytes,
            "the release took {} of the buffer's {bytes} bytes off the footprint",
            before.saturating_sub(after)
        );
    }

    /// Private-buffer staging chunks are charged to the device store's own
    /// ledger entry for the duration of each blit and released immediately
    /// after, so a store without room refuses the staging allocation instead
    /// of silently allocating uncharged host RAM.
    #[test]
    fn private_staging_is_charged_to_the_device_store_and_released() {
        use seismic_memory::entry::Tally;
        let Ok(handle) = DeviceHandle::system_default() else {
            return;
        };
        let store = Arc::new(Tally::unbounded(seismic_topology::StoreIdentity::HostRam));
        let device = MetalDevice::open(handle, store.clone()).unwrap();
        let baseline = store.held();
        let private = device
            .allocate_private(64 << 10, &crate::test_support::entry(), seismic_memory::SlabFill::Zeroed)
            .unwrap();
        let payload = vec![0x5Au8; 64 << 10];
        private.write_bytes(0, &payload).unwrap();
        assert_eq!(store.held(), baseline, "staging charge is released after write");
        let mut back = vec![0u8; payload.len()];
        private.read_bytes(0, &mut back).unwrap();
        assert_eq!(back, payload);
        assert_eq!(store.held(), baseline, "staging charge is released after read");
        // Bound the device store to its baseline so no staging chunk fits.
        store.bound(baseline);
        assert!(matches!(
            private.write_bytes(0, &payload),
            Err(ExecutionError::AllocationCapacity { .. })
        ));
        assert!(matches!(
            private.read_bytes(0, &mut back),
            Err(ExecutionError::AllocationCapacity { .. })
        ));
        assert_eq!(store.held(), baseline);
    }

    /// This process's physical footprint now, read live.
    fn footprint() -> u64 {
        seismic_native_target::process::macos::task_vm_info()
            .expect("task_info TASK_VM_INFO")
            .phys_footprint
    }

    #[test]
    fn mmap_enomem_maps_to_device_refused_and_other_errno_to_allocation_failed() {
        use seismic_compiler::errors::RefusalReading;

        let oom = super::classify_mmap_error(
            4096,
            std::io::Error::from_raw_os_error(libc::ENOMEM),
            |required| ExecutionError::DeviceRefused { required, reading: RefusalReading::Host },
        );
        assert!(oom.capacity_refusal().is_some(), "{oom:?}");
        assert!(matches!(oom, ExecutionError::DeviceRefused { required: 4096, .. }));

        let other = super::classify_mmap_error(
            4096,
            std::io::Error::from_raw_os_error(libc::EINVAL),
            |required| ExecutionError::DeviceRefused { required, reading: RefusalReading::Host },
        );
        assert!(other.capacity_refusal().is_none(), "{other:?}");
        assert!(matches!(other, ExecutionError::AllocationFailed(_)));
    }

    #[test]
    fn command_buffer_oom_maps_to_device_refused_and_other_error_to_fallback() {
        use objc2_foundation::NSError;
        use objc2_metal::{MTLCommandBufferError, MTLCommandBufferErrorDomain};
        use seismic_compiler::errors::RefusalReading;

        crate::pooled(|| {
            let domain = unsafe { MTLCommandBufferErrorDomain };
            let oom_ns = unsafe {
                NSError::errorWithDomain_code_userInfo(
                    domain,
                    MTLCommandBufferError::OutOfMemory.0 as isize,
                    None,
                )
            };
            let oom = super::classify_command_buffer_error(
                &oom_ns,
                || ExecutionError::DeviceRefused { required: 0, reading: RefusalReading::Host },
                ExecutionError::SubmissionFailed,
            );
            assert!(oom.capacity_refusal().is_some(), "{oom:?}");
            assert!(matches!(oom, ExecutionError::DeviceRefused { required: 0, .. }));

            let timeout_ns = unsafe {
                NSError::errorWithDomain_code_userInfo(
                    domain,
                    MTLCommandBufferError::Timeout.0 as isize,
                    None,
                )
            };
            let timeout = super::classify_command_buffer_error(
                &timeout_ns,
                || ExecutionError::DeviceRefused { required: 0, reading: RefusalReading::Host },
                ExecutionError::SubmissionFailed,
            );
            assert!(timeout.capacity_refusal().is_none(), "{timeout:?}");
            assert!(matches!(timeout, ExecutionError::SubmissionFailed(_)));
        });
    }
}
