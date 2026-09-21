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

    /// Cheap catalog memory figure. This is Metal's directly queried maximum
    /// buffer length, not a profiled or inferred capacity.
    pub fn memory_bytes(&self) -> u64 {
        self.device.maxBufferLength() as u64
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

    pub(crate) fn queue_retained(&self) -> Retained<ProtocolObject<dyn MTLCommandQueue>> {
        self.queue.clone()
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
        Ok(MetalBuffer { buffer, len: bytes })
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
