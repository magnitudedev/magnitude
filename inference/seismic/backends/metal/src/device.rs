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
use std::fmt;
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

    fn add(self: &Arc<Self>, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let mut wired = self.lock();
        crate::pooled(|| {
            wired.set.addAllocation(ProtocolObject::from_ref(buffer));
            wired.set.commit();
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

    fn remove(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let renewer = {
            let mut wired = self.lock();
            crate::pooled(|| {
                wired.set.removeAllocation(ProtocolObject::from_ref(buffer));
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

/// A buffer's membership of its device's residency set. The buffer is wired
/// from the commit that adds it until its last handle is dropped, so the
/// kernel never compresses or swaps it while macOS honors the set's renewed
/// residency.
struct WiredAllocation {
    residency: Arc<Residency>,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for WiredAllocation {}
unsafe impl Sync for WiredAllocation {}

impl WiredAllocation {
    fn wire(residency: &Arc<Residency>, buffer: &Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        residency.add(buffer);
        Self {
            residency: residency.clone(),
            buffer: buffer.clone(),
        }
    }
}

impl Drop for WiredAllocation {
    fn drop(&mut self) {
        self.residency.remove(&self.buffer);
    }
}

/// One shared-storage Metal buffer.
#[derive(Clone)]
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: u64,
    _host_owner: Option<Arc<dyn Any + Send + Sync>>,
    /// Absent for a read-only host mapping, whose pages stay file-backed.
    _wired: Option<Arc<WiredAllocation>>,
    read_only: bool,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// GPU virtual address used by slab address tables on macOS 15 and later.
    pub fn gpu_address(&self) -> u64 {
        self.buffer.gpuAddress()
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Host view of `[offset, offset + len)`. Precondition (FFI wrapper,
    /// §13.3.3): the range lies inside the buffer.
    fn host_range(&self, offset: u64, len: usize) -> *mut u8 {
        let end = offset
            .checked_add(len as u64)
            .filter(|end| *end <= self.len);
        assert!(
            end.is_some(),
            "MetalBuffer host access [{offset}, +{len}) exceeds the {}-byte buffer",
            self.len
        );
        // Shared storage mode: `contents()` is host-visible for the buffer's
        // whole lifetime.
        unsafe {
            self.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset as usize)
        }
    }

    pub(crate) fn write_bytes(&self, offset: u64, bytes: &[u8]) {
        assert!(
            !self.read_only,
            "cannot write to a read-only mapped Metal buffer"
        );
        let destination = self.host_range(offset, bytes.len());
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len()) };
    }

    pub(crate) fn read_bytes(&self, offset: u64, into: &mut [u8]) {
        let source = self.host_range(offset, into.len());
        unsafe { std::ptr::copy_nonoverlapping(source, into.as_mut_ptr(), into.len()) };
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
        Ok(Self {
            handle,
            queue,
            residency: Residency::new(residency),
        })
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
    pub(crate) fn allocate_bytes(&self, bytes: u64) -> Result<MetalBuffer, ExecutionError> {
        let length = usize::try_from(bytes.max(1)).map_err(|_| {
            ExecutionError::AllocationFailed(format!("{bytes} bytes exceed the host address space"))
        })?;
        let buffer = crate::pooled(|| {
            self.handle
                .raw()
                .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        })
        .ok_or_else(|| {
            ExecutionError::AllocationFailed(format!("Metal refused a {bytes}-byte buffer"))
        })?;
        let wired = WiredAllocation::wire(&self.residency, &buffer);
        Ok(MetalBuffer {
            buffer,
            len: bytes,
            _host_owner: None,
            _wired: Some(Arc::new(wired)),
            read_only: false,
        })
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
            read_only: true,
        })
    }
}

impl DeviceService<Metal> for MetalDevice {
    type Buffer = MetalBuffer;

    fn allocate(&self, bytes: u64, _alignment: u64) -> Result<Self::Buffer, ExecutionError> {
        // The profile advertises the alignment returned by
        // `heapBufferSizeAndAlignWithLength:options:` for this same shared
        // buffer class; direct device buffers satisfy that requirement.
        self.allocate_bytes(bytes)
    }

    fn write(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), ExecutionError> {
        if buffer.read_only {
            return Err(ExecutionError::SubmissionFailed(
                "write to a read-only mapped Metal buffer".into(),
            ));
        }
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
        buffer.len
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
        assert!(device.write(&buffer, 0, &[1]).is_err());
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
                set.containsAllocation(ProtocolObject::from_ref(buffer.raw())),
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
