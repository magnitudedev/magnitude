//! Transfers: copies between host memory and a device's storage (either
//! way) that run beside its compute, queued in batches the host numbers and
//! observes complete.
//!
//! A transfer is what moves bytes between storage the host addresses and
//! storage it does not. Where the host addresses both ends it copies in
//! place and forms no transfer. Per backend:
//!
//! | Backend | Transfer |
//! |---|---|
//! | CUDA | a stream of its own, one copy a run by where its ends lie (`cuMemcpyHtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoDAsync`; host ends pinned and mapped), then `cuStreamWriteValue32` of the batch's value into a mapped word (`seismic_cuda::transfer`) |
//! | CPU | the device's memory is the host's: each run is copied on the calling thread, complete on return |
//! | Metal | none: every storage is host-addressed (shared or wired), so the host writes in place ([`TransferError::InPlace`]) |
//! | Vulkan | `vkCmdCopyBuffer` regions on the compute family's second queue, signalling a timeline semaphore of its own with the batch's value (`seismic_vulkan::transfer`); none on a family of one queue, which reserves no device storage |
//!
//! A batch is a sequence of runs and signals: a signal sets a 4-byte word of
//! the device's storage to its value once every run of the batch before it
//! has landed, so device work can wait for one record of a batch by an
//! equality on its word (CUDA: `cuStreamWriteValue32` on the copy stream;
//! Vulkan: `vkCmdFillBuffer` behind a barrier over the copies before it;
//! CPU: a release store after the copies).
//!
//! Batches complete in the order they were queued. The caller orders a
//! transfer against device work: it queues one only into bytes no queued
//! work reaches, and lets work read them only once it has seen the batch
//! complete (a host gate it answers after [`TransferInner::completed`]).
//! The source's bytes stay untouched until then. Each batch is recorded as
//! a device use of its runs' sources and destinations, on the device's
//! transfer queue: host access to them, and their storage's release or
//! unbacking, wait for it as for compute work.

use crate::api::device::DeviceInner;
use crate::api::tensor::TensorInner;
use crate::backends::OpenedKind;
use crate::completion::DeviceCompletion;
use crate::driver::{typed_buffer, DeviceQueue};
use seismic_compiler::errors::ExecutionError;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Why a device forms no transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransferError {
    /// The backend's every storage is host-addressed: the host writes in
    /// place, and nothing transfers.
    InPlace,
    /// A run's or a signal's tensors are another device's, or lie outside
    /// the bytes their views reach.
    Run(String),
    Execution(ExecutionError),
}

impl fmt::Display for TransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InPlace => formatter.write_str("the device's storage is host-addressed: the host writes it in place"),
            Self::Run(why) => write!(formatter, "a transfer run is invalid: {why}"),
            Self::Execution(error) => write!(formatter, "a transfer failed: {error}"),
        }
    }
}

impl std::error::Error for TransferError {}

/// One copy of a batch: `bytes` bytes of `source` from `source_offset`
/// into `destination` from `destination_offset`, both offsets within the
/// bytes the tensors' views reach: host memory into the device's storage,
/// or back.
pub struct TransferRun<'a> {
    pub source: &'a TensorInner,
    pub source_offset: u64,
    pub destination: &'a TensorInner,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// One step of a batch, in order: a copy, or a signal: the i32 word at byte
/// `offset` of `word`'s view (this device's storage) set to `value` once
/// every run of the batch before it has landed.
pub enum TransferItem<'a> {
    Run(TransferRun<'a>),
    Signal { word: &'a TensorInner, offset: u64, value: i32 },
}

enum TransferKind {
    /// The device's memory is the host's: each batch is copied as queued;
    /// `done` is the last batch's value.
    Cpu { done: AtomicU64 },
    Cuda(seismic_cuda::transfer::TransferQueue),
    #[cfg(not(target_os = "macos"))]
    Vulkan(seismic_vulkan::transfer::TransferQueue),
}

/// The queue batches run on, and how their completion is observed. Shared
/// with the fences its batches leave on the allocations they reach.
struct Queue {
    kind: TransferKind,
}

impl Queue {
    /// Whether batch `value` is complete, or why it never will be: the
    /// queue faulted or the device is lost. Each backend reports that only
    /// once no copy runs any more: its queue has drained, or its context or
    /// device is lost.
    fn completed(&self, value: u64) -> Result<bool, ExecutionError> {
        match &self.kind {
            TransferKind::Cpu { done } => Ok(done.load(Ordering::Acquire) >= value),
            TransferKind::Cuda(queue) => queue.completed(value as u32),
            #[cfg(not(target_os = "macos"))]
            TransferKind::Vulkan(queue) => queue.reached(value),
        }
    }

    /// Wait at most `timeout` for batch `value`, sleeping until the device
    /// completes it: a CPU batch completed as it was queued.
    fn wait(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        match &self.kind {
            TransferKind::Cpu { .. } => self.completed(value),
            TransferKind::Cuda(queue) => queue.wait(value as u32, timeout),
            #[cfg(not(target_os = "macos"))]
            TransferKind::Vulkan(queue) => queue.wait(value, timeout),
        }
    }
}

/// A queued batch, as the allocations its runs read and write observe it:
/// complete, or faulted once no copy of it runs ([`Queue::completed`]).
struct BatchFence {
    queue: Arc<Queue>,
    value: u64,
}

impl DeviceCompletion for BatchFence {
    fn completed(&self) -> Result<bool, ExecutionError> {
        self.queue.completed(self.value)
    }

    fn ended(&self) -> Result<(), ExecutionError> {
        while !self.queue.wait(self.value, Duration::from_secs(1))? {}
        Ok(())
    }
}

/// One device's transfer queue.
pub struct TransferInner {
    device: Arc<DeviceInner>,
    queue: Arc<Queue>,
    /// The value the next batch takes: batches are numbered here alone,
    /// every backend's queue signalling the value it is given.
    next: AtomicU64,
    /// Queues batches one at a time, so values reach the device in order.
    issuing: Mutex<()>,
}

impl TransferInner {
    /// The transfer queue of `device`.
    pub fn new(device: &Arc<DeviceInner>) -> Result<Self, TransferError> {
        let kind = match &device.kind {
            OpenedKind::Cpu(_) => TransferKind::Cpu { done: AtomicU64::new(0) },
            OpenedKind::Cuda(opened) => TransferKind::Cuda(
                seismic_cuda::transfer::TransferQueue::new(opened.service()).map_err(TransferError::Execution)?,
            ),
            #[cfg(target_os = "macos")]
            OpenedKind::Metal(_) => return Err(TransferError::InPlace),
            #[cfg(not(target_os = "macos"))]
            OpenedKind::Vulkan(opened) => TransferKind::Vulkan(
                seismic_vulkan::transfer::TransferQueue::new(opened.service()).map_err(TransferError::Execution)?,
            ),
        };
        Ok(Self {
            device: device.clone(),
            queue: Arc::new(Queue { kind }),
            next: AtomicU64::new(1),
            issuing: Mutex::new(()),
        })
    }

    /// Queue `items` as one batch: the value it completes at
    /// ([`Self::completed`]). An invalid batch is refused before anything is
    /// queued; once queued, a failure is the queue's (`completed` reports
    /// it). The caller queues it only once the device work that read the
    /// bytes it overwrites has ended (the module documentation), so no batch
    /// waits on device work. The
    /// batch is a device use of every run's source (read) and destination
    /// (written) and every signal's word (written): host access to them,
    /// and their storage's release, wait for it.
    pub fn copy(&self, items: &[TransferItem<'_>]) -> Result<u64, TransferError> {
        for item in items {
            match item {
                TransferItem::Run(run) => self.check(run)?,
                TransferItem::Signal { word, offset, .. } => self.check_signal(word, *offset)?,
            }
        }
        // Host access is excluded while the batch is queued, in access-group
        // order as native submissions take theirs.
        let access = crate::native::merge_access(items.iter().flat_map(|item| match item {
            TransferItem::Run(run) => {
                vec![(run.source.allocation().clone(), false), (run.destination.allocation().clone(), true)]
            }
            TransferItem::Signal { word, .. } => vec![(word.allocation().clone(), true)],
        }));
        let permits = access
            .iter()
            .map(|(allocation, write)| allocation.acquire_for_device(*write))
            .collect::<Vec<_>>();
        let _issuing = self.issuing.lock().expect("transfer queue lock is never poisoned");
        // A batch takes its value once queued: a refused one takes none.
        let value = self.next.load(Ordering::Relaxed);
        match &self.queue.kind {
            TransferKind::Cpu { done } => {
                // Every item's addresses first: a batch is refused before any
                // of it is copied.
                enum Step {
                    Copy(*const u8, *mut u8, usize),
                    Signal(*mut u8, i32),
                }
                let at = |tensor: &TensorInner, offset: u64| {
                    let base = tensor
                        .allocation()
                        .storage()
                        .host_address()
                        .ok_or_else(|| TransferError::Run("a CPU tensor without host memory".into()))?;
                    // SAFETY: within the tensor's view, checked above.
                    Ok::<_, TransferError>(unsafe { base.as_ptr().add((tensor.byte_offset() + offset) as usize) })
                };
                let steps = items
                    .iter()
                    .map(|item| match item {
                        TransferItem::Run(run) => Ok(Step::Copy(
                            at(run.source, run.source_offset)?,
                            at(run.destination, run.destination_offset)?,
                            run.bytes as usize,
                        )),
                        TransferItem::Signal { word, offset, value } => Ok(Step::Signal(at(word, *offset)?, *value)),
                    })
                    .collect::<Result<Vec<_>, TransferError>>()?;
                for step in steps {
                    match step {
                        // SAFETY: both ranges lie in live views (checked);
                        // the caller keeps any other access off them
                        // meanwhile.
                        Step::Copy(source, destination, bytes) => unsafe { std::ptr::copy(source, destination, bytes) },
                        // SAFETY: an aligned word of a live view (checked),
                        // which device work reads atomically.
                        Step::Signal(word, value) => unsafe {
                            std::sync::atomic::AtomicI32::from_ptr(word.cast()).store(value, Ordering::Release)
                        },
                    }
                }
                done.store(value, Ordering::Release);
            }
            TransferKind::Cuda(queue) => {
                type Cuda = seismic_cuda::Cuda;
                type Executor = seismic_cuda::Executor;
                let items = items
                    .iter()
                    .map(|item| match item {
                        TransferItem::Run(run) => seismic_cuda::transfer::TransferItem::Run(seismic_cuda::transfer::TransferRun {
                            source: typed_buffer::<Cuda, Executor>(run.source.allocation()),
                            source_offset: run.source.byte_offset() + run.source_offset,
                            destination: typed_buffer::<Cuda, Executor>(run.destination.allocation()),
                            destination_offset: run.destination.byte_offset() + run.destination_offset,
                            bytes: run.bytes,
                        }),
                        TransferItem::Signal { word, offset, value } => seismic_cuda::transfer::TransferItem::Signal {
                            word: typed_buffer::<Cuda, Executor>(word.allocation()),
                            offset: word.byte_offset() + offset,
                            value: *value as u32,
                        },
                    })
                    .collect::<Vec<_>>();
                // The word holds a value's low 32 bits, compared cyclically.
                queue.copy(&items, value as u32).map_err(TransferError::Execution)?;
            }
            #[cfg(not(target_os = "macos"))]
            TransferKind::Vulkan(queue) => {
                use crate::backends::vulkan_transfer_end;
                let items = items
                    .iter()
                    .map(|item| match item {
                        TransferItem::Run(run) => seismic_vulkan::transfer::TransferItem::Run(seismic_vulkan::transfer::TransferRun {
                            source: vulkan_transfer_end(run.source.allocation(), run.source.byte_offset() + run.source_offset),
                            destination: vulkan_transfer_end(
                                run.destination.allocation(),
                                run.destination.byte_offset() + run.destination_offset,
                            ),
                            bytes: run.bytes,
                        }),
                        TransferItem::Signal { word, offset, value } => seismic_vulkan::transfer::TransferItem::Signal {
                            word: vulkan_transfer_end(word.allocation(), word.byte_offset() + offset),
                            value: *value as u32,
                        },
                    })
                    .collect::<Vec<_>>();
                queue.copy(&items, value).map_err(TransferError::Execution)?;
            }
        }
        self.next.store(value + 1, Ordering::Relaxed);
        let fence: Arc<dyn DeviceCompletion> = Arc::new(BatchFence { queue: self.queue.clone(), value });
        for (allocation, write) in &access {
            allocation.record_device_use(DeviceQueue::Transfer, fence.clone(), *write);
        }
        drop(permits);
        Ok(value)
    }

    /// A run copies between live views of this device's tensors, each
    /// within the bytes its view reaches (on chunked storage its records as
    /// packed, which run past its canonical bytes).
    fn check(&self, run: &TransferRun<'_>) -> Result<(), TransferError> {
        let within =
            |tensor: &TensorInner, offset: u64| offset.checked_add(run.bytes).is_some_and(|end| end <= tensor.reached_bytes());
        if !within(run.source, run.source_offset) || !within(run.destination, run.destination_offset) {
            return Err(TransferError::Run("a run lies outside its tensors".into()));
        }
        if !Arc::ptr_eq(run.source.device(), &self.device) || !Arc::ptr_eq(run.destination.device(), &self.device) {
            return Err(TransferError::Run("a run's tensors are another device's".into()));
        }
        Ok(())
    }

    /// A signal's word is an aligned i32 of a live view of this device's.
    fn check_signal(&self, word: &TensorInner, offset: u64) -> Result<(), TransferError> {
        if offset.checked_add(4).is_none_or(|end| end > word.reached_bytes()) {
            return Err(TransferError::Run("a signal's word lies outside its tensor".into()));
        }
        if (word.byte_offset() + offset) % 4 != 0 {
            return Err(TransferError::Run("a signal's word is not 4-byte aligned".into()));
        }
        if !Arc::ptr_eq(word.device(), &self.device) {
            return Err(TransferError::Run("a signal's word is another device's".into()));
        }
        Ok(())
    }

    /// Whether the batch that took `value` is complete, its bytes visible
    /// to device work begun after this returns true; why no batch can
    /// complete (a copy faulted, or the device is lost).
    pub fn completed(&self, value: u64) -> Result<bool, TransferError> {
        self.queue.completed(value).map_err(TransferError::Execution)
    }

    /// Wait at most `timeout` for the batch that took `value`; whether it
    /// completed, or why none can. The thread sleeps until the batch
    /// completes, whatever other queues of the device still run.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<bool, TransferError> {
        self.queue.wait(value, timeout).map_err(TransferError::Execution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Arc<DeviceInner> {
        let devices = crate::devices::Devices::discover(&[]).unwrap();
        devices.open_backend(seismic_lang::registry::BackendName::Cpu).unwrap()
    }

    #[test]
    fn a_cpu_batch_copies_its_runs_and_completes_on_return() {
        let device = cpu();
        let transfer = TransferInner::new(&device).unwrap();
        let tensor = |bytes: &[u8]| {
            let representation = seismic_lang::registry::dense(seismic_lang::types::DType::I32);
            crate::api::tensor::TensorInner::from_host(&device, representation, &[bytes.len() as u64 / 4], bytes).unwrap()
        };
        let source = tensor(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let destination = tensor(&[0; 8]);
        let words = tensor(&[0; 8]);
        let value = transfer
            .copy(
                &[
                    TransferItem::Run(TransferRun { source: &source, source_offset: 0, destination: &destination, destination_offset: 4, bytes: 2 }),
                    TransferItem::Signal { word: &words, offset: 4, value: 5 },
                    TransferItem::Run(TransferRun { source: &source, source_offset: 6, destination: &destination, destination_offset: 0, bytes: 2 }),
                ],
            )
            .unwrap();
        assert!(transfer.completed(value).unwrap());
        assert!(!transfer.completed(value + 1).unwrap());
        // The batch is one fence, recorded on its source and destination.
        assert_eq!(Arc::strong_count(&transfer.queue), 2);
        assert_eq!(destination.read_to_host().unwrap(), [7, 8, 0, 0, 1, 2, 0, 0]);
        assert_eq!(words.read_to_host().unwrap(), [0, 0, 0, 0, 5, 0, 0, 0], "the signal's word holds its value");
        // A run past its tensor, or a signal past or within a word, is
        // refused, queuing nothing: a batch with a valid run before it
        // writes nothing either.
        let past = TransferRun { source: &source, source_offset: 7, destination: &destination, destination_offset: 0, bytes: 2 };
        assert!(matches!(transfer.copy(&[TransferItem::Run(past)]), Err(TransferError::Run(_))));
        for offset in [6, 8] {
            let first = TransferRun { source: &source, source_offset: 0, destination: &destination, destination_offset: 0, bytes: 2 };
            let signal = TransferItem::Signal { word: &words, offset, value: 1 };
            assert!(matches!(transfer.copy(&[TransferItem::Run(first), signal]), Err(TransferError::Run(_))));
        }
        assert_eq!(destination.read_to_host().unwrap(), [7, 8, 0, 0, 1, 2, 0, 0], "a refused batch wrote nothing");
        assert!(!transfer.completed(value + 1).unwrap(), "a refused batch takes no value");
    }
}
