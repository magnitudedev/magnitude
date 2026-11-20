//! Host blocks a device reads in place over its link: one plain host
//! allocation each ([`HostBlock`], in large pages where the host has them),
//! backed whole and registered whole for the device (`cuMemHostRegister`),
//! unregistered as it drops, before its pages return to the OS.
//!
//! `cuMemHostUnregister` waits for the device's work and holds back its copy
//! queue meanwhile: beside a decode flight parked on a pull, unregistering
//! 16 MiB took 2.1 s and the park gave up (two RTX 3090 hosts). With no
//! device work running, 14 MiB in 2 MiB pieces unregistered in 0.99 ms and
//! 1 GiB in 48.3 ms, 22 GB/s (RTX 4070, driver 595.99.02). A block drops only after the completion of the
//! last flight that read it, so its own work is never what the
//! unregistration waits for; another load's work on the device in the same
//! process is, and no data is at risk. No lock of the backend's is held
//! across the driver call, and no enqueue takes one ([`crate::enqueue`]).
//!
//! Its pages lie where the host's first touch places them: the process may
//! not set their node in a container (`mbind` was refused on two RTX 3090
//! hosts), and the node made no difference to the link (a 2990WX's block on
//! node 2 of a device on node 0, and an EPYC 7313's over nodes 0-3 of one on
//! node 1, read as fast as host-NUMA allocations of the device's node).

use crate::driver::{Context, DriverError};
use seismic_compiler::errors::{ExecutionError, RefusalReading};
use seismic_memory::entry::{Backed, Entry, Wiring};
use seismic_memory::{HostBlock, PageSize};
use std::ffi::{c_int, c_uint, c_void};
use std::mem::ManuallyDrop;
use std::sync::Arc;

/// `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`: pinned for
/// every context, mapped for the device.
const HOST_REGISTER: c_uint = 1 | 2;

/// `CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`.
const HOST_POINTER: c_int = 91;

fn host_allocation_error(bytes: u64, failure: seismic_memory::AllocationFailure) -> ExecutionError {
    match failure {
        seismic_memory::AllocationFailure::OutOfMemory => ExecutionError::DeviceRefused {
            required: bytes,
            reading: RefusalReading::Host,
        },
        other => ExecutionError::AllocationFailed(format!("host block pages: {other}")),
    }
}

/// One host block, registered whole for the device: zero when made, read
/// and written by the host in place and by the device at [`Self::device`],
/// its exact bytes charged to its entry while it lives. Backed, it unmaps
/// and frees its charge in one change.
pub(crate) struct RegisteredBlock(Backed<Registered>);

impl std::ops::Deref for RegisteredBlock {
    type Target = Registered;

    fn deref(&self) -> &Registered {
        &self.0
    }
}

pub(crate) struct Registered {
    // Unregistered in `drop` before the block unmaps; intentionally leaked
    // if unregistration fails so CUDA-pinned pages are never unmapped.
    block: ManuallyDrop<HostBlock>,
    device: u64,
    context: Arc<Context>,
}

impl RegisteredBlock {
    /// At least `bytes`, in whole registration granules
    /// (`seismic_memory::registration_granule`, a large page where the
    /// host has them), charged to `entry` before it is backed.
    pub fn new(context: &Arc<Context>, entry: &Arc<dyn Entry>, bytes: u64) -> Result<Self, ExecutionError> {
        let granule = seismic_memory::registration_granule().map_err(|error| host_allocation_error(bytes, error))?;
        let whole = bytes
            .max(1)
            .checked_next_multiple_of(granule)
            .ok_or_else(|| host_allocation_error(bytes, seismic_memory::AllocationFailure::Unrepresentable))?;
        let whole_usize = usize::try_from(whole)
            .map_err(|_| host_allocation_error(bytes, seismic_memory::AllocationFailure::Unrepresentable))?;
        let failed = |error: DriverError| crate::executor::allocation_error(whole, || RefusalReading::Host)(error);
        Backed::take(entry, whole, Wiring::Pageable, |_| {
            let block = HostBlock::allocate(whole, PageSize::Large).map_err(|error| host_allocation_error(whole, error))?;
            assert_eq!(block.bytes() as u64, whole, "invariant violated: a block of whole registration granules is that many bytes");
            let driver = &context.driver;
            let _current = context.enter().map_err(failed)?;
            let base: *mut c_void = block.base().as_ptr().cast();
            {
                let _changing = context.changing_memory();
                unsafe { driver.check((driver.host_register)(base, whole_usize, HOST_REGISTER), "host register") }.map_err(failed)?;
            }
            // Owned from here: if the device-pointer query fails, `Registered::drop`
            // unregisters before dropping `block`, or leaks `block` on unregistration failure.
            let mut registered = Registered {
                block: ManuallyDrop::new(block),
                device: 0,
                context: context.clone(),
            };
            registered.device = if driver.attribute(HOST_POINTER, context.ordinal()).map_err(failed)? != 0 {
                base as u64
            } else {
                let mut device = 0u64;
                unsafe {
                    driver.check(
                        (driver.host_device_pointer)(&mut device, base, 0),
                        "host device pointer",
                    )
                }
                .map_err(failed)?;
                device
            };
            Ok::<_, ExecutionError>(registered)
        })
        .map(Self)
    }
}

impl Registered {
    /// The address the device reads it at.
    pub fn device(&self) -> u64 {
        self.device
    }

    /// Its first byte, which the host reads and writes in place.
    pub fn host(&self) -> std::ptr::NonNull<u8> {
        self.block.base()
    }

    pub fn bytes(&self) -> usize {
        self.block.bytes()
    }
}

/// Try to unregister the host block at `base`. Returns `true` only when
/// `cuMemHostUnregister` succeeds; on failure (for example on context loss),
/// logs a warning and returns `false` so the caller does not unmap pages
/// the driver may still pin.
fn try_unregister(context: &Context, base: *mut c_void) -> bool {
    let Ok(_current) = context.enter() else {
        tracing::warn!(
            "CUDA registered host block drop: failed to enter context; skipping unregistration and leaking host pages"
        );
        return false;
    };
    let _changing = context.changing_memory();
    let driver = &context.driver;
    if let Err(error) = unsafe { driver.check((driver.host_unregister)(base), "host unregister") } {
        tracing::warn!("CUDA registered host block drop: {error}; leaking host pages");
        return false;
    }
    true
}

/// Drops `block` if and only if `unregistered` is `true`; otherwise
/// intentionally leaves it un-dropped to avoid unmapping CUDA-pinned pages.
unsafe fn drop_block_if_unregistered<T>(block: &mut ManuallyDrop<T>, unregistered: bool) {
    if unregistered {
        unsafe {
            ManuallyDrop::drop(block);
        }
    }
}

impl Drop for Registered {
    /// Unregistered before its block unmaps; on unregistration failure,
    /// `self.block` is intentionally leaked without panicking.
    fn drop(&mut self) {
        let unregistered = try_unregister(&self.context, self.block.base().as_ptr().cast());
        unsafe {
            drop_block_if_unregistered(&mut self.block, unregistered);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{drop_block_if_unregistered, host_allocation_error};
    use seismic_compiler::errors::{ExecutionError, RefusalReading};
    use seismic_memory::AllocationFailure;
    use std::mem::ManuallyDrop;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn host_oom_maps_to_device_refused_with_capacity_refusal() {
        let error = host_allocation_error(4 << 20, AllocationFailure::OutOfMemory);
        assert!(error.capacity_refusal().is_some(), "{error:?}");
        assert!(matches!(
            error,
            ExecutionError::DeviceRefused {
                required,
                reading: RefusalReading::Host,
            } if required == (4 << 20)
        ));
    }

    #[test]
    fn non_oom_host_failure_maps_to_allocation_failed() {
        let error = host_allocation_error(4 << 20, AllocationFailure::Unrepresentable);
        assert!(error.capacity_refusal().is_none(), "{error:?}");
        assert!(matches!(error, ExecutionError::AllocationFailed(_)));
    }

    #[test]
    fn drop_block_only_drops_when_unregistered() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut failed = ManuallyDrop::new(DropCounter(drops.clone()));
        unsafe { drop_block_if_unregistered(&mut failed, false) };
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        let mut succeeded = ManuallyDrop::new(DropCounter(drops.clone()));
        unsafe { drop_block_if_unregistered(&mut succeeded, true) };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
