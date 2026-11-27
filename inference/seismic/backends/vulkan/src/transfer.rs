//! Copies between host and device storage, either way, on the device's
//! second queue, beside the work its first runs
//! (`seismic_runtime::native::transfer`): each batch one command buffer of
//! `vkCmdCopyBuffer` regions, and of each signal a `vkCmdFillBuffer` of its
//! word behind a barrier over the copies before it, submitted to the second
//! queue (`DeviceInfo::side_family`, which pool blocks' zero fills share),
//! signalling the queue's own timeline semaphore with the batch's value.
//! The semaphore's value is the last batch complete.
//!
//! Ordering against the first queue's work, both ways, runs through the
//! host:
//! - The caller queues a batch into bytes only once the first queue's work
//!   that read them has completed (`seismic_runtime::native::transfer`), so
//!   no batch waits on that work.
//! - The host learns a batch is complete only by a semaphore wait on its
//!   value ([`TransferQueue::reached`] waits with no timeout). The batch's
//!   signal makes its writes (behind the batch's barrier) available, and the
//!   host's wait happens-after it. Work that reads them is released only by
//!   the host after that wait, through a semaphore of its own: a park's
//!   answer (the host signals the park timeline the parked submission
//!   waits on), or the submission of a step the host queues once it mapped
//!   the copied record. That submission's semaphore wait makes the writes
//!   visible to the work after it: transfer signal → host wait → host
//!   signal or submission → the waiting submission is one happens-before
//!   chain. No kernel waits on the transfer queue itself.

use crate::device::{barrier, call, Device};
use crate::Buffer;
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use std::any::Any;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// One end of a run: a buffer's bytes from an offset.
pub struct TransferEnd<'a>(pub &'a Buffer, pub u64);

impl TransferEnd<'_> {
    /// The `(buffer, offset)` of `bytes` bytes from the end's byte 0, which
    /// lie within its buffer.
    pub(crate) fn span(&self, bytes: u64) -> Result<(vk::Buffer, u64), ExecutionError> {
        let TransferEnd(buffer, offset) = *self;
        let range = buffer.range();
        if offset.checked_add(bytes).is_none_or(|end| end > range.bytes()) {
            return Err(ExecutionError::SubmissionFailed("a transfer run lies outside its buffer".into()));
        }
        Ok((range.raw(), range.offset() + offset))
    }
}

/// One copy of a batch: `bytes` bytes from `source` into `destination`.
pub struct TransferRun<'a> {
    pub source: TransferEnd<'a>,
    pub destination: TransferEnd<'a>,
    pub bytes: u64,
}

/// One step of a batch, in order: a copy, or a signal: the 4-byte word at
/// `word` set to `value` once every copy of the batch before it has landed.
pub enum TransferItem<'a> {
    Run(TransferRun<'a>),
    Signal { word: TransferEnd<'a>, value: u32 },
}

/// A recorded step of a batch.
enum Command {
    Copy(vk::Buffer, vk::Buffer, vk::BufferCopy),
    Fill(vk::Buffer, u64, u32),
}

type WaitOwner = Arc<dyn Any + Send + Sync>;

struct State {
    pool: vk::CommandPool,
    /// Command buffers and foreign wait-semaphore owners of batches submitted,
    /// retained until `self.semaphore` reaches `value`, and free command buffers.
    flying: Vec<(u64, vk::CommandBuffer, Vec<WaitOwner>)>,
    free: Vec<vk::CommandBuffer>,
}

/// A device's copy queue and its timeline.
pub struct TransferQueue {
    device: Device,
    semaphore: vk::Semaphore,
    state: Mutex<State>,
}

impl TransferQueue {
    /// A queue on `device`'s second queue, whose timeline holds 0; why the
    /// device has none (a family of one queue).
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        if !device.has_side_queue() {
            return Err(ExecutionError::SubmissionFailed(
                "the device has no second queue: no transfer runs beside its work".into(),
            ));
        }
        let raw = &device.inner.device;
        let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let semaphore =
            call(unsafe { raw.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None) }, "vkCreateSemaphore")?;
        let pool = match call(
            unsafe {
                raw.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(device.inner.info.side_family.expect("a second queue has a family"))
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
            },
            "vkCreateCommandPool",
        ) {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { raw.destroy_semaphore(semaphore, None) };
                return Err(error);
            }
        };
        Ok(Self {
            device: device.clone(),
            semaphore,
            state: Mutex::new(State { pool, flying: Vec::new(), free: Vec::new() }),
        })
    }

    /// The timeline semaphore signalled as each batch completes, for GPU-side
    /// cross-queue waits (`DirectBatch::wait_device_semaphore`).
    pub fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// Queue `items` as one batch, which raises the timeline to `value`
    /// once every run has landed and every signal is written: the caller
    /// numbers batches, each above the last. A batch that is invalid or
    /// fails is refused before it is submitted. The caller copies only into
    /// bytes no queued work reaches, and keeps the sources' bytes until
    /// then.
    pub fn copy(&self, items: &[TransferItem<'_>], value: u64) -> Result<(), ExecutionError> {
        self.copy_after(items, value, &[], Vec::new())
    }

    /// [`Self::copy`], waiting on device-signalled `waits` (at `TRANSFER`)
    /// before executing the copies (e.g. waiting for compute readers to retire
    /// before overwriting a scratch slot), keeping `wait_owners` alive until
    /// `self.semaphore` reaches `value`.
    pub fn copy_after(
        &self,
        items: &[TransferItem<'_>],
        value: u64,
        waits: &[(vk::Semaphore, u64)],
        wait_owners: Vec<WaitOwner>,
    ) -> Result<(), ExecutionError> {
        let mut regions: Vec<Command> = Vec::new();
        for item in items {
            let run = match item {
                TransferItem::Run(run) => run,
                TransferItem::Signal { word, value } => {
                    let (buffer, offset) = word.span(4)?;
                    if offset % 4 != 0 {
                        return Err(ExecutionError::SubmissionFailed("a transfer signal's word is not 4-byte aligned".into()));
                    }
                    regions.push(Command::Fill(buffer, offset, *value));
                    continue;
                }
            };
            let ((source, source_offset), (destination, destination_offset)) =
                (run.source.span(run.bytes)?, run.destination.span(run.bytes)?);
            if run.bytes > 0 {
                regions.push(Command::Copy(
                    source,
                    destination,
                    vk::BufferCopy { src_offset: source_offset, dst_offset: destination_offset, size: run.bytes },
                ));
            }
        }
        let raw = &self.device.inner.device;
        let mut state = self.state.lock().expect("transfer lock is never poisoned");
        self.reclaim(&mut state)?;
        let command = match state.free.pop() {
            Some(command) => command,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(state.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                call(unsafe { raw.allocate_command_buffers(&info) }, "vkAllocateCommandBuffers")?[0]
            }
        };
        let recorded = unsafe {
            call(raw.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()), "vkResetCommandBuffer")
                .and_then(|()| {
                    call(
                        raw.begin_command_buffer(
                            command,
                            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        ),
                        "vkBeginCommandBuffer",
                    )
                })
                .and_then(|()| {
                    for step in &regions {
                        match *step {
                            Command::Copy(source, destination, region) => {
                                raw.cmd_copy_buffer(command, source, destination, std::slice::from_ref(&region))
                            }
                            // A signal follows every copy before it.
                            Command::Fill(buffer, offset, value) => {
                                barrier(
                                    raw,
                                    command,
                                    vk::PipelineStageFlags2::TRANSFER,
                                    vk::AccessFlags2::TRANSFER_WRITE,
                                    vk::PipelineStageFlags2::TRANSFER,
                                    vk::AccessFlags2::TRANSFER_WRITE,
                                );
                                raw.cmd_fill_buffer(command, buffer, offset, 4, value);
                            }
                        }
                    }
                    // The batch's writes are made available to later work and
                    // to the host, which reads a copy into host storage once
                    // its wait on the batch's value returns.
                    barrier(
                        raw,
                        command,
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST,
                        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_READ,
                    );
                    call(raw.end_command_buffer(command), "vkEndCommandBuffer")
                })
        };
        if let Err(error) = recorded {
            state.free.push(command);
            return Err(error);
        }
        let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
        let wait_infos: Vec<vk::SemaphoreSubmitInfo<'_>> = waits
            .iter()
            .map(|&(semaphore, wait_value)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(wait_value)
                    .stage_mask(vk::PipelineStageFlags2::TRANSFER)
            })
            .collect();
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals);
        if let Err(error) = self.device.submit_side(&[submit]) {
            state.free.push(command);
            return Err(error);
        }
        state.flying.push((value, command, wait_owners));
        Ok(())
    }

    /// The command buffers of completed batches, free again, and their
    /// foreign wait-semaphore owners released once the GPU has retired them.
    fn reclaim(&self, state: &mut State) -> Result<(), ExecutionError> {
        let reached = self
            .device
            .inner
            .vk("vkGetSemaphoreCounterValue", |device| unsafe {
                device.get_semaphore_counter_value(self.semaphore)
            })?;
        let (done, flying): (Vec<_>, Vec<_>) = state.flying.drain(..).partition(|(value, _, _)| *value <= reached);
        state.flying = flying;
        state.free.extend(done.into_iter().map(|(_, command, _owners)| command));
        Ok(())
    }

    /// Whether batch `value` is complete, by a semaphore wait that returns
    /// at once (the module documentation: the host happens-after its
    /// signal); why the device can report none (it is lost).
    pub fn reached(&self, value: u64) -> Result<bool, ExecutionError> {
        self.wait(value, Duration::ZERO)
    }

    /// Wait at most `timeout` for batch `value`; whether it completed, or
    /// why the device can make no progress.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<bool, ExecutionError> {
        self.device.inner.wait_semaphore(self.semaphore, value, timeout)
    }
}

impl Drop for TransferQueue {
    /// Every batch queued completes before its command buffers, pool and
    /// semaphore are freed, and before the storage it copies may be. Where
    /// the wait fails (the device is lost) they are never freed: a batch may
    /// still use them.
    fn drop(&mut self) {
        if self.device.inner.is_lost() {
            return;
        }
        let state = self.state.get_mut().expect("transfer lock is never poisoned");
        let last = state.flying.iter().map(|(value, _, _)| *value).max().unwrap_or(0);
        let semaphores = [self.semaphore];
        let values = [last];
        let raw = &self.device.inner.device;
        // Every compute submission waiting on `self.semaphore` retains an
        // `Arc<TransferQueue>` in `Retired::wait_owners` until it retires on
        // the GPU, so by the time `Drop` runs no compute submission can hold a
        // pending wait on `self.semaphore` — we never block on the compute
        // queue here (avoiding deadlock if dropped on a gate-signalling thread).
        unsafe {
            if raw
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values),
                    u64::MAX,
                )
                .is_ok()
            {
                raw.destroy_command_pool(state.pool, None);
                raw.destroy_semaphore(self.semaphore, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On every device with a second queue, a batch copies upload storage
    /// into device storage byte for byte, each run signalling a word of its
    /// own, and completes at its value; a device without one forms no
    /// transfer queue. It prints the rate of a batch of the whole buffer.
    #[test]
    fn a_batch_lands_its_runs_and_signals() {
        for description in crate::test_devices() {
            let device = Device::open(description.info.uuid, crate::unbounded_stores(&description.info), &Default::default()).expect("a device meeting the floor opens");
            let queue = TransferQueue::new(&device);
            assert_eq!(queue.is_ok(), device.has_side_queue(), "{}: a transfer queue exactly where a side queue is", description.info.name);
            let Ok(queue) = queue else {
                continue;
            };
            let own = device.own();
            let bytes = 8u64 << 20;
            let destination = device.allocate(bytes, 256, &own).expect("device storage");
            let source = device.allocate_upload(bytes, 256, &own).expect("upload storage");
            let pattern = (0..bytes).map(|at| (at % 251) as u8 ^ (at >> 20) as u8).collect::<Vec<_>>();
            device.write(&source, 0, &pattern).unwrap();
            // The source's second half to the destination's start and its
            // first half after, each run signalling a word of its own.
            let half = bytes / 2;
            let words = device.allocate_upload(8, 256, &own).expect("signal words");
            device.write(&words, 0, &[0; 8]).unwrap();
            queue
                .copy(&[
                    TransferItem::Run(TransferRun {
                        source: TransferEnd(&source, half),
                        destination: TransferEnd(&destination, 0),
                        bytes: half,
                    }),
                    TransferItem::Signal { word: TransferEnd(&words, 0), value: 7 },
                    TransferItem::Run(TransferRun {
                        source: TransferEnd(&source, 0),
                        destination: TransferEnd(&destination, half),
                        bytes: half,
                    }),
                    TransferItem::Signal { word: TransferEnd(&words, 4), value: 9 },
                ], 1)
                .unwrap();
            assert!(queue.wait(1, Duration::from_secs(10)).unwrap(), "{}: the batch completes", description.info.name);
            let mut signalled = [0u8; 8];
            device.read(&words, 0, &mut signalled).unwrap();
            assert_eq!(signalled, [7, 0, 0, 0, 9, 0, 0, 0], "{}: each signal is written", description.info.name);
            let mut landed = vec![0u8; bytes as usize];
            device.read(&destination, 0, &mut landed).unwrap();
            let expected = pattern[half as usize..].iter().chain(&pattern[..half as usize]).copied().collect::<Vec<_>>();
            assert!(landed == expected, "{}: the runs land byte for byte", description.info.name);
            // Back to host storage: the destination's first 4 KiB.
            let back = device.allocate_upload(4096, 256, &own).expect("upload storage");
            let run = TransferRun { source: TransferEnd(&destination, 0), destination: TransferEnd(&back, 0), bytes: 4096 };
            queue.copy(&[TransferItem::Run(run)], 2).unwrap();
            assert!(queue.wait(2, Duration::from_secs(10)).unwrap());
            let mut returned = vec![0u8; 4096];
            device.read(&back, 0, &mut returned).unwrap();
            assert!(returned == expected[..4096], "{}: the device's bytes return to the host", description.info.name);
            let started = std::time::Instant::now();
            let run = TransferRun { source: TransferEnd(&source, 0), destination: TransferEnd(&destination, 0), bytes };
            queue.copy(&[TransferItem::Run(run)], 3).unwrap();
            assert!(queue.wait(3, Duration::from_secs(10)).unwrap());
            eprintln!(
                "{}: transfer of {} MiB at {:.2} GB/s",
                description.info.name,
                bytes >> 20,
                bytes as f64 / started.elapsed().as_secs_f64() / 1e9
            );
        }
    }

    /// GPU-side cross-queue timeline synchronization on devices with a side
    /// queue:
    /// 1. Producer publication before consumer read: `TransferQueue::copy` on
    ///    `side_family` writes `mid`, and a compute-queue `DirectBatch`
    ///    immediately waits on `(queue.semaphore(), 1)` via
    ///    `wait_device_semaphore` (no host wait!) and copies `mid -> dst`.
    /// 2. Consumer retirement before slot reuse: `TransferQueue::copy_after` on
    ///    `side_family` waits on `compute_sub.timeline_wait()` before
    ///    overwriting `mid` with a second pattern, leaving `dst` intact with
    ///    the first pattern and `mid` holding the second.
    #[test]
    fn gpu_side_queue_semaphore_orders_producer_before_consumer_and_retires_before_reuse() {
        for description in crate::test_devices() {
            let device = Device::open(
                description.info.uuid,
                crate::unbounded_stores(&description.info),
                &Default::default(),
            )
            .expect("a device meeting the floor opens");
            if !device.has_side_queue() {
                continue;
            }
            let queue = std::sync::Arc::new(TransferQueue::new(&device).expect("side queue opens transfer queue"));
            let own = device.own();
            let bytes = 4u64 << 20;
            let src1 = device.allocate_upload(bytes, 256, &own).expect("src1 upload");
            let src2 = device.allocate_upload(bytes, 256, &own).expect("src2 upload");
            let mid = device.allocate(bytes, 256, &own).expect("mid device");
            let dst = device.allocate(bytes, 256, &own).expect("dst device");
            let pat1: Vec<u8> = (0..bytes).map(|i| ((i * 7 + 3) & 0xff) as u8).collect();
            let pat2: Vec<u8> = (0..bytes).map(|i| ((i * 13 + 91) & 0xff) as u8).collect();
            device.write(&src1, 0, &pat1).unwrap();
            device.write(&src2, 0, &pat2).unwrap();

            // 1. Producer on side_family copies src1 -> mid (timeline value 1).
            queue
                .copy(
                    &[TransferItem::Run(TransferRun {
                        source: TransferEnd(&src1, 0),
                        destination: TransferEnd(&mid, 0),
                        bytes,
                    })],
                    1,
                )
                .unwrap();

            // 2. Consumer on compute queue waits on (queue.semaphore(), 1) on GPU
            //    (zero host wait!) and copies mid -> dst.
            let mut batch = crate::direct::DirectBatch::unmeasured(&device).unwrap();
            batch.wait_device_semaphore_owned(
                queue.semaphore(),
                1,
                std::sync::Arc::clone(&queue) as std::sync::Arc<dyn std::any::Any + Send + Sync>,
            );
            batch
                .transfer(&[(TransferEnd(&mid, 0), TransferEnd(&dst, 0), bytes)], &[])
                .unwrap();
            let compute_sub = batch.commit().unwrap();

            // 3. Producer reuses `mid` on side_family ONLY after compute reader
            //    retires (`compute_sub.timeline_wait()`).
            queue
                .copy_after(
                    &[TransferItem::Run(TransferRun {
                        source: TransferEnd(&src2, 0),
                        destination: TransferEnd(&mid, 0),
                        bytes,
                    })],
                    2,
                    &[compute_sub.timeline_wait()],
                    Vec::new(),
                )
                .unwrap();

            assert!(queue.wait(2, Duration::from_secs(10)).unwrap());
            compute_sub.finish().unwrap();

            let mut got_dst = vec![0u8; bytes as usize];
            let mut got_mid = vec![0u8; bytes as usize];
            device.read(&dst, 0, &mut got_dst).unwrap();
            device.read(&mid, 0, &mut got_mid).unwrap();
            assert!(got_dst == pat1, "{}: compute read saw exact pat1 before reuse", description.info.name);
            assert!(got_mid == pat2, "{}: mid holds pat2 after consumer retired", description.info.name);
        }
    }
}
