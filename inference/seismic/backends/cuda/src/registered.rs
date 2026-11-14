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
use std::sync::Arc;

/// `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`: pinned for
/// every context, mapped for the device.
const HOST_REGISTER: c_uint = 1 | 2;

/// `CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`.
const HOST_POINTER: c_int = 91;

fn host_pages(failure: seismic_memory::AllocationFailure) -> DriverError {
    DriverError {
        operation: "host block pages",
        code: 0,
        description: failure.to_string(),
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
    // Unregistered in `drop` before the block unmaps.
    block: HostBlock,
    device: u64,
    context: Arc<Context>,
}

impl RegisteredBlock {
    /// At least `bytes`, in whole registration granules
    /// (`seismic_memory::registration_granule`, a large page where the
    /// host has them), charged to `entry` before it is backed.
    pub fn new(context: &Arc<Context>, entry: &Arc<dyn Entry>, bytes: u64) -> Result<Self, ExecutionError> {
        let failed = |error: DriverError| crate::executor::allocation_error(bytes, || RefusalReading::Host)(error);
        let granule = seismic_memory::registration_granule().map_err(|error| failed(host_pages(error)))?;
        let whole = bytes
            .max(1)
            .checked_next_multiple_of(granule)
            .ok_or_else(|| failed(host_pages(seismic_memory::AllocationFailure::Unrepresentable)))?;
        Backed::take(entry, whole, Wiring::Pageable, |_| {
            let block = HostBlock::allocate(whole, PageSize::Large).map_err(|error| failed(host_pages(error)))?;
            assert_eq!(block.bytes() as u64, whole, "invariant violated: a block of whole registration granules is that many bytes");
            let driver = &context.driver;
            let _current = context.enter().map_err(failed)?;
            let base: *mut c_void = block.base().as_ptr().cast();
            {
                let _changing = context.changing_memory();
                unsafe { driver.check((driver.host_register)(base, whole as usize, HOST_REGISTER), "host register") }.map_err(failed)?;
            }
            let device = (|| {
                if driver.attribute(HOST_POINTER, context.ordinal())? != 0 {
                    return Ok(base as u64);
                }
                let mut device = 0u64;
                unsafe { driver.check((driver.host_device_pointer)(&mut device, base, 0), "host device pointer")? };
                Ok(device)
            })();
            let device = device.map_err(|error| {
                unregister(context, base);
                failed(error)
            })?;
            Ok::<_, ExecutionError>(Registered { block, device, context: context.clone() })
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

/// Unregister the block at `base`. A block no work uses that will not
/// unregister is a driver fault it cannot outlive: unmapping it would free
/// pages the driver still pins.
fn unregister(context: &Context, base: *mut c_void) {
    let driver = &context.driver;
    let _current = context.enter().unwrap_or_else(|error| panic!("a registered host block's context enters: {error}"));
    let _changing = context.changing_memory();
    unsafe { driver.check((driver.host_unregister)(base), "host unregister") }
        .unwrap_or_else(|error| panic!("a registered host block no work uses unregisters: {error}"));
}

impl Drop for Registered {
    /// Unregistered before its block unmaps (the fields drop after this).
    fn drop(&mut self) {
        unregister(&self.context, self.block.base().as_ptr().cast());
    }
}
