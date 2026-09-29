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
    MTLResourceOptions,
};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::DeviceService;
use std::fmt;
use std::{any::Any, ptr::NonNull, sync::Arc};

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
        MTLCopyAllDevices()
            .into_iter()
            .map(|device| Self { device })
            .collect()
    }

    /// The system default Metal device.
    pub fn system_default() -> Result<Self, TargetError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| TargetError::DeviceUnavailable("no Metal device".into()))?;
        Ok(Self { device })
    }

    pub fn registry_id(&self) -> u64 {
        self.device.registryID()
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
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

/// One shared-storage Metal buffer.
#[derive(Clone)]
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: u64,
    _host_owner: Option<Arc<dyn Any + Send + Sync>>,
    read_only: bool,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
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
}

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    pub fn open(handle: DeviceHandle) -> Result<Self, TargetError> {
        // Compilation is shared by every native entry formed on this device.
        // Metal defaults to a small compiler pool; allow the system to scale
        // compilation concurrency to the machine before forming any kernels.
        handle.raw().setShouldMaximizeConcurrentCompilation(true);
        let queue = handle.raw().newCommandQueue().ok_or_else(|| {
            TargetError::DeviceUnavailable("could not create a Metal command queue".into())
        })?;
        Ok(Self { handle, queue })
    }

    pub fn handle(&self) -> &DeviceHandle {
        &self.handle
    }

    pub(crate) fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Allocates a shared buffer of `bytes`. Metal rejects a zero-length
    /// buffer, so a zero-byte allocation reserves one byte while reporting
    /// its requested length.
    pub(crate) fn allocate_bytes(&self, bytes: u64) -> Result<MetalBuffer, ExecutionError> {
        let length = usize::try_from(bytes.max(1)).map_err(|_| {
            ExecutionError::AllocationFailed(format!("{bytes} bytes exceed the host address space"))
        })?;
        let buffer = self
            .handle
            .raw()
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                ExecutionError::AllocationFailed(format!("Metal refused a {bytes}-byte buffer"))
            })?;
        Ok(MetalBuffer {
            buffer,
            len: bytes,
            _host_owner: None,
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
        let buffer = unsafe {
            self.handle
                .raw()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer.cast(),
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| {
            ExecutionError::AllocationFailed("Metal refused a read-only host mapping".into())
        })?;
        Ok(MetalBuffer {
            buffer,
            len: length as u64,
            _host_owner: Some(owner),
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
        drop(buffer);
        assert!(weak.upgrade().is_none());
    }
}
