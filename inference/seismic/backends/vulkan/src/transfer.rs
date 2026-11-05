//! Copies between host and device storage, either way, on the device's
//! second queue, beside the work its first runs
//! (`seismic_runtime::native::transfer`): each batch one command buffer of
//! `vkCmdCopyBuffer` regions, and of each signal a `vkCmdFillBuffer` of its
//! word behind a barrier over the copies before it, submitted to the second
//! queue (`Facts::side_family`, which reservations' zero fills share),
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
use crate::memory::Range;
use crate::{Buffer, ReservedBuffer};
use ash::vk;
use seismic_compiler::errors::ExecutionError;
use std::sync::Mutex;
use std::time::Duration;

/// One end of a run: bytes of a buffer, or of a reservation's committed
/// bytes (split at its chunks).
pub enum TransferEnd<'a> {
    Buffer(&'a Buffer, u64),
    Reserved(&'a ReservedBuffer, u64),
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

struct State {
    pool: vk::CommandPool,
    /// Command buffers of batches submitted, by value, and free ones.
    flying: Vec<(u64, vk::CommandBuffer)>,
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
                        .queue_family_index(device.inner.facts.side_family.expect("a second queue has a family"))
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

    /// Each `(buffer, offset)` share of `bytes` bytes from `end`'s byte 0.
    fn spans(end: &TransferEnd<'_>, bytes: u64) -> Result<Vec<(vk::Buffer, u64, u64)>, ExecutionError> {
        let within = |range: &Range, offset: u64| {
            if offset.checked_add(bytes).is_none_or(|end| end > range.bytes()) {
                return Err(ExecutionError::SubmissionFailed("a transfer run lies outside its buffer".into()));
            }
            Ok(vec![(range.raw(), range.offset() + offset, bytes)])
        };
        match end {
            TransferEnd::Buffer(buffer, offset) => within(buffer.range(), *offset),
            TransferEnd::Reserved(reserved, offset) => {
                if offset.checked_add(bytes).is_none_or(|end| end > reserved.len()) {
                    return Err(ExecutionError::SubmissionFailed("a transfer run lies outside its reservation".into()));
                }
                let mut spans = Vec::new();
                reserved.spans(*offset, bytes as usize, |chunk, at, _, len| {
                    spans.push((chunk.range().raw(), chunk.range().offset() + at, len as u64));
                    Ok(())
                })?;
                Ok(spans)
            }
        }
    }

    /// Queue `items` as one batch, which raises the timeline to `value`
    /// once every run has landed and every signal is written: the caller
    /// numbers batches, each above the last. A batch that is invalid or
    /// fails is refused before it is submitted. The caller copies only into
    /// bytes no queued work reaches, and keeps the sources' bytes until
    /// then.
    pub fn copy(&self, items: &[TransferItem<'_>], value: u64) -> Result<(), ExecutionError> {
        let mut regions: Vec<Command> = Vec::new();
        for item in items {
            let run = match item {
                TransferItem::Run(run) => run,
                TransferItem::Signal { word, value } => {
                    let spans = Self::spans(word, 4)?;
                    let &[(buffer, offset, 4)] = spans.as_slice() else {
                        return Err(ExecutionError::SubmissionFailed("a transfer signal's word spans chunks".into()));
                    };
                    if offset % 4 != 0 {
                        return Err(ExecutionError::SubmissionFailed("a transfer signal's word is not 4-byte aligned".into()));
                    }
                    regions.push(Command::Fill(buffer, offset, *value));
                    continue;
                }
            };
            let (sources, destinations) = (Self::spans(&run.source, run.bytes)?, Self::spans(&run.destination, run.bytes)?);
            // Pair the two sides' spans over the run's bytes.
            let (mut s, mut d) = (sources.into_iter().peekable(), destinations.into_iter().peekable());
            let (mut source, mut destination) = (s.next(), d.next());
            while let (Some((sb, so, sl)), Some((db, dob, dl))) = (source, destination) {
                let size = sl.min(dl);
                if size > 0 {
                    regions.push(Command::Copy(sb, db, vk::BufferCopy { src_offset: so, dst_offset: dob, size }));
                }
                source = if sl > size { Some((sb, so + size, sl - size)) } else { s.next() };
                destination = if dl > size { Some((db, dob + size, dl - size)) } else { d.next() };
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
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals);
        if let Err(error) = self.device.submit_side(&[submit]) {
            state.free.push(command);
            return Err(error);
        }
        state.flying.push((value, command));
        Ok(())
    }

    /// The command buffers of completed batches, free again.
    fn reclaim(&self, state: &mut State) -> Result<(), ExecutionError> {
        let reached = call(
            unsafe { self.device.inner.device.get_semaphore_counter_value(self.semaphore) },
            "vkGetSemaphoreCounterValue",
        )?;
        let (done, flying): (Vec<_>, Vec<_>) = state.flying.drain(..).partition(|&(value, _)| value <= reached);
        state.flying = flying;
        state.free.extend(done.into_iter().map(|(_, command)| command));
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
        let semaphores = [self.semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        let nanoseconds = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        match unsafe { self.device.inner.device.wait_semaphores(&info, nanoseconds) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => call(Err(error), "vkWaitSemaphores"),
        }
    }
}

impl Drop for TransferQueue {
    /// Every batch queued completes before its command buffers, pool and
    /// semaphore are freed, and before the storage it copies may be. Where
    /// the wait fails (the device is lost) they are never freed: a batch may
    /// still use them.
    fn drop(&mut self) {
        let state = self.state.get_mut().expect("transfer lock is never poisoned");
        let last = state.flying.iter().map(|&(value, _)| value).max().unwrap_or(0);
        let semaphores = [self.semaphore];
        let values = [last];
        let raw = &self.device.inner.device;
        unsafe {
            if raw.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values), u64::MAX).is_ok() {
                raw.destroy_command_pool(state.pool, None);
                raw.destroy_semaphore(self.semaphore, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On every device that reserves device storage, a batch copies upload
    /// storage into a reservation's chunks byte for byte, the runs crossing
    /// chunk boundaries, and completes at its value; a device without a
    /// second queue forms no transfer queue. It prints the
    /// rate of a batch of the whole reservation.
    #[test]
    fn a_batch_lands_its_runs_across_a_reservations_chunks() {
        for description in crate::test_devices() {
            let device = Device::open(description.facts.uuid, crate::test_accounts(&description.facts)).expect("a device meeting the floor opens");
            let queue = TransferQueue::new(&device);
            assert_eq!(queue.is_ok(), device.has_side_queue(), "{}: a transfer queue exactly where a side queue is", description.facts.name);
            // A device whose memory is the host's reserves without a second
            // queue, so it may transfer nothing.
            let (Some(form), Ok(queue)) =
                (device.facts().reservation(0).expect("reservation capability query"), queue)
            else {
                continue;
            };
            let granule = form.granule;
            let bytes = 4 * granule;
            let (reserved, _resizer) =
                device.allocate_reserved(&[0], bytes, bytes, granule, granule / 4).expect("a reservation");
            let source = device.allocate_upload(bytes, 256, None).expect("upload storage");
            let pattern = (0..bytes).map(|at| (at % 251) as u8 ^ (at / granule) as u8).collect::<Vec<_>>();
            device.write(&source, 0, &pattern).unwrap();
            // The source's second half to the reservation's start and its
            // first half after, each across two chunks, each run signalling a
            // word of its own.
            let half = bytes / 2;
            let words = device.allocate_upload(8, 256, None).expect("signal words");
            device.write(&words, 0, &[0; 8]).unwrap();
            queue
                .copy(&[
                    TransferItem::Run(TransferRun {
                        source: TransferEnd::Buffer(&source, half),
                        destination: TransferEnd::Reserved(&reserved, 0),
                        bytes: half,
                    }),
                    TransferItem::Signal { word: TransferEnd::Buffer(&words, 0), value: 7 },
                    TransferItem::Run(TransferRun {
                        source: TransferEnd::Buffer(&source, 0),
                        destination: TransferEnd::Reserved(&reserved, half),
                        bytes: half,
                    }),
                    TransferItem::Signal { word: TransferEnd::Buffer(&words, 4), value: 9 },
                ], 1)
                .unwrap();
            assert!(queue.wait(1, Duration::from_secs(10)).unwrap(), "{}: the batch completes", description.facts.name);
            let mut signalled = [0u8; 8];
            device.read(&words, 0, &mut signalled).unwrap();
            assert_eq!(signalled, [7, 0, 0, 0, 9, 0, 0, 0], "{}: each signal is written", description.facts.name);
            let mut landed = vec![0u8; bytes as usize];
            reserved.read(0, &mut landed).unwrap();
            let expected = pattern[half as usize..].iter().chain(&pattern[..half as usize]).copied().collect::<Vec<_>>();
            assert!(landed == expected, "{}: the runs land byte for byte", description.facts.name);
            // Back to host storage: the reservation's first 4 KiB.
            let back = device.allocate_upload(4096, 256, None).expect("upload storage");
            let run = TransferRun { source: TransferEnd::Reserved(&reserved, 0), destination: TransferEnd::Buffer(&back, 0), bytes: 4096 };
            queue.copy(&[TransferItem::Run(run)], 2).unwrap();
            assert!(queue.wait(2, Duration::from_secs(10)).unwrap());
            let mut returned = vec![0u8; 4096];
            device.read(&back, 0, &mut returned).unwrap();
            assert!(returned == expected[..4096], "{}: the device's bytes return to the host", description.facts.name);
            let started = std::time::Instant::now();
            let run = TransferRun { source: TransferEnd::Buffer(&source, 0), destination: TransferEnd::Reserved(&reserved, 0), bytes };
            queue.copy(&[TransferItem::Run(run)], 3).unwrap();
            assert!(queue.wait(3, Duration::from_secs(10)).unwrap());
            eprintln!(
                "{}: transfer of {} MiB at {:.2} GB/s",
                description.facts.name,
                bytes >> 20,
                bytes as f64 / started.elapsed().as_secs_f64() / 1e9
            );
        }
    }
}
