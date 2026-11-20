//! Ordered moves: byte copies between a device's allocations and 8-byte
//! word writes into them, run as device work in submission order, after the
//! work already submitted that uses their hazard domain and before the work
//! submitted after, so their submitter spends only the enqueue. A granule
//! pool's vacate is one: slabs copied into another block's granules and the
//! stores' tables repointed to them, beside the flights in flight
//! (`crate::pool`).
//!
//! | Backend | Moves |
//! |---|---|
//! | CUDA | the compute stream, after the fill stream's work they follow by what they touch (`Allocation::queue_order`): a `cuMemcpyDtoDAsync` per copy, two `cuMemsetD32Async` per word, the value carried in the call |
//! | Vulkan | one queue submission: a `vkCmdCopyBuffer` per copy and a `vkCmdUpdateBuffer` per word, the value carried in the command buffer, between barriers (`seismic_vulkan::direct::DirectBatch::transfer`) |
//! | CPU | its native work completes as it is submitted, so none is in flight: copied and written on the calling thread, as its native work runs |
//! | Metal | none formed: every storage is host-addressed, and the host copies and writes once the work using the domain has ended (host write access), the one backend whose moves wait for it |
//!
//! Moves are recorded as a writing use of their domain on the compute
//! queue: host access to it, the release of what lies in it, and native
//! work of the other queue order after them as after any native work.

use super::{IssuedWork, NativeQueue};
use crate::api::device::DeviceInner;
use crate::backends::OpenedKind;
use crate::completion::{Committed, DeviceCompletion, RouteSubmission};
use crate::driver::{Allocation, DeviceQueue};
use seismic_compiler::errors::ExecutionError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// One step of ordered moves ([`submit`]), each in its moves' hazard
/// domain: a move's copies come before its words.
pub(crate) enum MoveStep<'a> {
    /// `bytes` bytes of `source` from `source_offset` into `destination`
    /// from `destination_offset`.
    Copy { source: &'a Arc<Allocation>, source_offset: u64, destination: &'a Arc<Allocation>, destination_offset: u64, bytes: u64 },
    /// The 8-byte-aligned word at byte `offset` of `destination` set to
    /// `value`, whole: no reader ordered by the domain sees it half
    /// written.
    Word { destination: &'a Arc<Allocation>, offset: u64, value: u64 },
}

impl MoveStep<'_> {
    /// Every byte it touches lies in its allocations, and a word is
    /// aligned; else an invariant violation.
    fn check(&self, domain: &Arc<Allocation>) {
        let within = |allocation: &Arc<Allocation>, offset: u64, bytes: u64| {
            assert!(
                allocation.shares_access_with(domain),
                "invariant violated: an ordered move touches an allocation outside its hazard domain"
            );
            assert!(
                offset.checked_add(bytes).is_some_and(|end| end <= allocation.bytes()),
                "invariant violated: an ordered move of {bytes} bytes at {offset} exceeds its {}-byte allocation",
                allocation.bytes()
            );
        };
        match *self {
            Self::Copy { source, source_offset, destination, destination_offset, bytes } => {
                within(source, source_offset, bytes);
                within(destination, destination_offset, bytes);
            }
            Self::Word { destination, offset, .. } => {
                within(destination, offset, 8);
                assert!(offset % 8 == 0, "invariant violated: an ordered word at {offset} is not 8-byte aligned");
            }
        }
    }

    /// Run on the host, which addresses the storage: a copy or a word store.
    ///
    /// # Safety
    /// Nothing else accesses the bytes it writes meanwhile, but readers of
    /// a word, which read it whole.
    unsafe fn run_on_host(&self) {
        let at = |allocation: &Allocation, offset: u64| {
            let base = allocation
                .storage()
                .host_address()
                .expect("invariant violated: host-run moves touch storage the host does not address");
            // SAFETY: within the allocation, checked by `check`.
            unsafe { base.as_ptr().add(offset as usize) }
        };
        match *self {
            // SAFETY: both ranges lie in live allocations (`check`); a
            // pool's moves copy between distinct granules.
            Self::Copy { source, source_offset, destination, destination_offset, bytes } => unsafe {
                std::ptr::copy(at(source, source_offset), at(destination, destination_offset), bytes as usize)
            },
            // SAFETY: an aligned word of a live allocation, which device
            // work reads whole.
            Self::Word { destination, offset, value } => unsafe {
                AtomicU64::from_ptr(at(destination, offset).cast()).store(value, Ordering::Release)
            },
        }
    }
}

/// Host write permit acquired before taking a store's slab lock on Metal,
/// where moves drain device work on the domain before copying on the host.
/// Non-Metal backends enqueue asynchronously and need no pre-acquired permit.
pub(crate) struct MovePermit {
    #[cfg(target_os = "macos")]
    host: Option<crate::driver::AllocationPermit>,
}

/// Acquire any host write permit needed before locking `Slots::slabs` for
/// [`submit_prepared`]: on Metal, drains device work on `domain` via
/// `domain.acquire(true)` outside `Slots::slabs` so a parked GPU kernel
/// waiting on `region_view` cannot deadlock. No-op on other backends.
pub(crate) fn prepare(device: &Arc<DeviceInner>, domain: &Arc<Allocation>) -> Result<MovePermit, ExecutionError> {
    #[cfg(target_os = "macos")]
    if matches!(device.kind, OpenedKind::Metal(_)) {
        return Ok(MovePermit { host: Some(domain.acquire(true)?) });
    }
    let _ = (device, domain);
    Ok(MovePermit {
        #[cfg(target_os = "macos")]
        host: None,
    })
}

/// Run `steps`, in order (every copy before every word), after the device
/// work submitted before that uses
/// `domain`'s hazard domain and before any submitted after (the module
/// documentation): every step's allocations lie in that domain. Never
/// waits for device work but on Metal. The moves' completion where they run
/// on the device, which every later use of the domain follows; none where
/// they completed on return. Refused, nothing run, where nothing could be
/// issued.
///
/// # Panics
/// Where issuing failed after part of the moves was issued (each backend's
/// encoder): that part runs on, and what the moves' callers hold no longer
/// describes the domain.
pub(crate) fn submit(
    device: &Arc<DeviceInner>,
    domain: &Arc<Allocation>,
    steps: &[MoveStep<'_>],
) -> Result<Option<Arc<dyn DeviceCompletion>>, ExecutionError> {
    let permit = prepare(device, domain)?;
    submit_prepared(device, domain, steps, permit)
}

/// Like [`submit`], using a [`MovePermit`] already obtained from [`prepare`].
pub(crate) fn submit_prepared(
    device: &Arc<DeviceInner>,
    domain: &Arc<Allocation>,
    steps: &[MoveStep<'_>],
    permit: MovePermit,
) -> Result<Option<Arc<dyn DeviceCompletion>>, ExecutionError> {
    device.ran_work();
    for step in steps {
        step.check(domain);
    }
    assert!(
        !steps.windows(2).any(|pair| matches!(pair, [MoveStep::Word { .. }, MoveStep::Copy { .. }])),
        "invariant violated: ordered moves' copies come before their words"
    );
    #[cfg(target_os = "macos")]
    if matches!(device.kind, OpenedKind::Metal(_)) {
        assert!(permit.host.is_some(), "invariant violated: Metal moves require a host write permit");
        for step in steps {
            // SAFETY: `permit.host` excludes every other access to the domain.
            unsafe { step.run_on_host() };
        }
        drop(permit);
        return Ok(None);
    }
    drop(permit);
    // Owns retirement until the moves are recorded, as a native
    // submission's: unwinding waits for issued work before releasing.
    let mut issued = IssuedWork(None);
    // Host access is excluded while the moves are encoded; device work is
    // not waited for.
    let permit = domain.acquire_for_device(true);
    let _order = device.native.order.lock().expect("native submission order lock is never poisoned");
    let queue = NativeQueue::Compute;
    let after = domain.queue_order(queue, true)?;
    let route = issued.0.insert(Arc::new(RouteSubmission::Committed(encode(device, &after, steps)?)));
    let completion: Arc<dyn DeviceCompletion> = route.clone();
    domain.record_device_use(DeviceQueue::Native(queue), completion.clone(), true);
    drop(permit);
    drop(issued.take());
    Ok(Some(completion))
}

/// Encode `steps` on the device's compute queue, after `after` (the other
/// native queue's work they follow): refused where none of them was
/// issued.
fn encode(device: &Arc<DeviceInner>, after: &[Arc<dyn DeviceCompletion>], steps: &[MoveStep<'_>]) -> Result<Committed, ExecutionError> {
    match &device.kind {
        OpenedKind::Cpu(_) => {
            assert!(after.is_empty(), "invariant violated: the CPU runs native work on one queue");
            let start = super::trace::host_seconds();
            for step in steps {
                // SAFETY: the device permit excludes host access, and the
                // CPU's native work, complete as submitted, runs none.
                unsafe { step.run_on_host() };
            }
            Ok(Committed::Cpu { outcome: Ok(()), interval: (start, super::trace::host_seconds()), launches: None })
        }
        OpenedKind::Cuda(opened) => super::cuda::encode_moves(opened.service(), after, steps, |allocation, offset| {
            device.allocation_address(allocation, offset)
        }),
        #[cfg(not(target_os = "macos"))]
        OpenedKind::Vulkan(opened) => {
            assert!(after.is_empty(), "invariant violated: Vulkan runs native work on one queue");
            super::vulkan::encode_moves(opened.service(), steps)
        }
        #[cfg(target_os = "macos")]
        OpenedKind::Metal(_) => unreachable!("Metal's moves run on the host"),
    }
}
