//! Copies between the host's pinned memory and the device's on the device's
//! copy engine, beside its compute (`seismic_runtime::native::transfer`): a
//! stream of their own, never the execution stream nor the context's
//! synchronous-read stream, so a copy waits for no launch and no
//! synchronous host read waits for a copy.
//!
//! A batch is queued in its order: one copy per run, by where its ends lie
//! (`cuMemcpyHtoDAsync` from pinned host memory the device maps,
//! `cuMemcpyDtoHAsync` into it, `cuMemcpyDtoDAsync` within the device's),
//! and per signal a `cuStreamWriteValue32` of its value into its word
//! (device memory a kernel may spin on), then `cuStreamWriteValue32` of the
//! batch's value into the queue's mapped completion word: each write follows
//! every copy queued before it (the default flags fence them), so a
//! signal's word holds its value once the runs before it have landed, and
//! the completion word reaches the batch's once every run of the batch, and
//! of every batch before it, has. The stream runs its batches in the order
//! they were queued, so the completion word's value is the last batch
//! complete.

use crate::buffer::Buffer;
use crate::direct::HostWord;
use crate::driver::Stream;
use crate::executor::Device;
use seismic_compiler::errors::ExecutionError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One copy of a batch: `bytes` bytes from `source` at `source_offset` to
/// `destination` at `destination_offset`, each pinned host memory the
/// device maps or the device's own, not both host memory.
pub struct TransferRun<'a> {
    pub source: &'a Buffer,
    pub source_offset: u64,
    pub destination: &'a Buffer,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// One step of a batch, in order: a copy, or a signal: the 4-byte word at
/// `offset` of `word` (device memory, or pinned host memory the device
/// maps) set to `value` once every run of the batch before it has landed.
pub enum TransferItem<'a> {
    Run(TransferRun<'a>),
    Signal { word: &'a Buffer, offset: u64, value: u32 },
}

/// Why a driver before API 11.7 has no transfer queue.
const PREDATES_STREAM_VALUES: &str =
    "the CUDA driver predates driver API 11.7 stream memory operations: no transfer signals its completion";

/// A device's copy queue and its completion word.
pub struct TransferQueue {
    device: Device,
    stream: Arc<Stream>,
    /// Dropped only once the stream is known to write it no more.
    done: std::mem::ManuallyDrop<HostWord>,
    /// Lock-free mirror of `fault.is_some()`, set under `fault` before waking
    /// waiters so `wait` and `completed` never block on `fault`.
    faulted: AtomicBool,
    /// Serializes `copy` enqueueing (`fault` check, `queue`, `wake_after`, and
    /// fault transition) and records why the queue can complete no batch from
    /// now after a driver call fails mid-enqueue. Never acquired on the
    /// lock-free `wait`/`completed` happy paths.
    fault: std::sync::Mutex<Option<String>>,
}

// The stream and word are driver objects usable from any thread with their
// context made current, which every call does.
unsafe impl Send for TransferQueue {}
unsafe impl Sync for TransferQueue {}

fn submission(error: crate::driver::DriverError) -> ExecutionError {
    ExecutionError::SubmissionFailed(error.to_string())
}

/// A batch's step as the stream queues it, checked: a copy by the driver
/// call its ends take, or a signal's word and value.
enum Checked {
    Upload { source: *const u8, destination: u64, bytes: usize },
    Download { source: u64, destination: *mut u8, bytes: usize },
    Within { source: u64, destination: u64, bytes: usize },
    Signal { word: u64, value: u32 },
}

impl TransferQueue {
    /// A queue on `device` whose completion word holds 0; refused where the
    /// driver writes no stream memory word (before driver API 11.7).
    pub fn new(device: &Device) -> Result<Self, ExecutionError> {
        if !device.context().driver.enqueue.stream_values() {
            return Err(ExecutionError::SubmissionFailed(PREDATES_STREAM_VALUES.into()));
        }
        Ok(Self {
            device: device.clone(),
            stream: Stream::new(device.context()).map_err(submission)?,
            done: std::mem::ManuallyDrop::new(HostWord::new(device)?),
            faulted: AtomicBool::new(false),
            fault: std::sync::Mutex::new(None),
        })
    }

    /// Queue `items` in order, then the completion word's raise to `value`
    /// (its low 32 bits, compared cyclically by [`Self::completed`]) once
    /// they and every batch queued before have landed. A batch that is
    /// invalid (a run between host memory or outside its buffers, a signal
    /// outside its buffer or unaligned), or one offered to a faulted queue,
    /// is refused before anything is queued; once queuing begins the batch
    /// is the queue's, and a driver call that fails then faults the queue
    /// ([`Self::completed`]). The caller keeps every run's bytes untouched
    /// until the batch is complete.
    pub fn copy(&self, items: &[TransferItem<'_>], value: u32) -> Result<(), ExecutionError> {
        let checked = items.iter().map(Self::check).collect::<Result<Vec<_>, _>>()?;
        let mut faulted_now = false;
        {
            let mut fault = self.fault.lock().expect("transfer queue lock is never poisoned");
            if let Some(reason) = fault.as_deref() {
                return Err(ExecutionError::SubmissionFailed(format!("the transfer queue faulted: {reason}")));
            }
            if let Err(error) = self.queue(&checked, value).map_err(submission).and_then(|()| self.done.wake_after(&self.stream)) {
                *fault = Some(error.to_string());
                self.faulted.store(true, Ordering::Release);
                faulted_now = true;
            }
        }
        if faulted_now {
            // Its waiters learn the fault now ([`Self::completed`]), with
            // `fault` already released and `faulted` published.
            self.done.wake();
        }
        Ok(())
    }

    /// Wait at most `timeout` for batch `value` ([`Self::completed`]),
    /// sleeping until the completion word's write wakes the waiter.
    pub fn wait(&self, value: u32, timeout: std::time::Duration) -> Result<bool, ExecutionError> {
        self.done.wait(|done| (done.wrapping_sub(value) as i32) >= 0 || self.faulted(), timeout);
        self.completed(value)
    }

    fn faulted(&self) -> bool {
        self.faulted.load(Ordering::Acquire)
    }

    /// A batch's item as the driver calls that queue it: a signal, or a
    /// run's copy.
    fn check(item: &TransferItem<'_>) -> Result<Checked, ExecutionError> {
        let refused = |why: &str| ExecutionError::SubmissionFailed(why.into());
        let run = match item {
            TransferItem::Run(run) => run,
            TransferItem::Signal { word, offset, value } => {
                if offset % 4 != 0 || offset.checked_add(4).is_none_or(|end| end > word.len()) {
                    return Err(refused("a transfer signal's word lies outside its buffer or is unaligned"));
                }
                return Ok(Checked::Signal { word: word.pointer() + offset, value: *value });
            }
        };
        let within = |buffer: &Buffer, offset: u64| offset.checked_add(run.bytes).is_some_and(|end| end <= buffer.len());
        if !within(run.source, run.source_offset) || !within(run.destination, run.destination_offset) {
            return Err(refused("a transfer run lies outside its buffers"));
        }
        let host = |bytes: u64| usize::try_from(bytes).map_err(|_| refused("a transfer run exceeds the host"));
        let (source_host, destination_host) = (run.source.host_address(), run.destination.host_address());
        if source_host.is_some() && destination_host.is_some() {
            return Err(refused("a transfer run between host memory: the host copies it"));
        }
        let bytes = host(run.bytes)?;
        let (source, destination) = (run.source.pointer() + run.source_offset, run.destination.pointer() + run.destination_offset);
        Ok(match (source_host, destination_host) {
            // SAFETY: within the source's mapped bytes, checked above.
            (Some(mapped), _) => Checked::Upload { source: unsafe { mapped.as_ptr().add(host(run.source_offset)?) }, destination, bytes },
            // SAFETY: within the destination's mapped bytes, checked above.
            (None, Some(mapped)) => {
                Checked::Download { source, destination: unsafe { mapped.as_ptr().add(host(run.destination_offset)?) }, bytes }
            }
            (None, None) => Checked::Within { source, destination, bytes },
        })
    }

    fn queue(&self, steps: &[Checked], value: u32) -> Result<(), crate::driver::DriverError> {
        let context = self.device.context();
        let _current = context.enter()?;
        let issue = context.enqueue();
        let driver = &context.driver;
        let stream = self.stream.raw();
        // The queue's making checked the driver writes stream words.
        let write = |stream, word, value| {
            unsafe { issue.write_value_32(stream, word, value, 0) }.ok_or_else(|| crate::driver::DriverError {
                operation: "transfer signal",
                code: 0,
                description: PREDATES_STREAM_VALUES.into(),
            })
        };
        for step in steps {
            let status = match *step {
                Checked::Upload { bytes: 0, .. } | Checked::Download { bytes: 0, .. } | Checked::Within { bytes: 0, .. } => continue,
                Checked::Upload { source, destination, bytes } => unsafe { issue.upload_async(destination, source.cast(), bytes, stream) },
                Checked::Download { source, destination, bytes } => unsafe {
                    issue.download_async(destination.cast(), source, bytes, stream)
                },
                Checked::Within { source, destination, bytes } => unsafe { issue.memcpy_device_async(destination, source, bytes, stream) },
                // CU_STREAM_WRITE_VALUE_DEFAULT: the write follows the
                // copies before it.
                Checked::Signal { word, value } => write(stream, word, value)?,
            };
            driver.check(status, "transfer copy")?;
        }
        // CU_STREAM_WRITE_VALUE_DEFAULT: the write follows the copies'.
        let status = write(stream, self.done.device_address(), value)?;
        driver.check(status, "transfer completion")
    }

    /// Whether batch `value` (its low 32 bits, compared cyclically) is
    /// complete, or why it never will be: the queue faulted (a driver call
    /// failed while a batch was queued, or a copy faulted, its error the
    /// stream's). A fault is reported once the stream has drained, or once
    /// the context is lost (the stream's query fails): either way no copy of
    /// any batch still writes.
    pub fn completed(&self, value: u32) -> Result<bool, ExecutionError> {
        let reached = || (self.done.load().wrapping_sub(value) as i32) >= 0;
        if !self.faulted() && reached() {
            return Ok(true);
        }
        if !self.stream.query().map_err(submission)? {
            return Ok(false);
        }
        if !self.faulted() && reached() {
            return Ok(true);
        }
        let fault = self.fault.lock().expect("transfer queue lock is never poisoned").clone();
        match fault {
            Some(fault) => Err(ExecutionError::SubmissionFailed(format!("the transfer queue faulted: {fault}"))),
            // The stream drained: the word holds the last value it wrote.
            None => Ok(reached()),
        }
    }
}

impl Drop for TransferQueue {
    /// Every queued copy and completion write ends before the word, and the
    /// storage the copies reach, may be freed. Where the stream cannot be
    /// waited for (the context is lost) the word is never freed: a write
    /// may still reach it.
    fn drop(&mut self) {
        if self.stream.synchronize().is_ok() {
            // SAFETY: dropped once, here; the stream writes it no more.
            unsafe { std::mem::ManuallyDrop::drop(&mut self.done) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{DirectBatch, DirectLaunch, DirectModule, TimelineAnchor};
    use std::time::{Duration, Instant};

    /// One block spins `cycles` clock cycles, then writes 1.
    const SPIN: &str = r#"
struct Words { unsigned long long cycles; };
extern "C" __global__ void spin(unsigned int* out, Words words, unsigned int* scalars) {
    long long start = clock64();
    while (clock64() - start < (long long)words.cycles) {}
    out[0] = 1;
}
"#;

    const BYTES: u64 = 256 << 20;

    /// Pinned host bytes holding `pattern(i)` at byte `i`, and a device
    /// block as large, as a pool's that an expert bank's records lie in.
    fn buffers(device: &Device, pattern: impl Fn(u64) -> u8) -> (Buffer, Buffer) {
        let source = device.allocate_mapped(BYTES, &crate::test_entry()).unwrap();
        let host = source.host_address().unwrap();
        for at in 0..BYTES {
            // SAFETY: within the mapping, which no device work reaches yet.
            unsafe { host.as_ptr().add(at as usize).write(pattern(at)) };
        }
        let destination = device.allocate_block(crate::topology::DEVICE_KIND, BYTES, &crate::test_entry(), seismic_memory::SlabFill::Zeroed).unwrap();
        (source, destination)
    }

    fn run(source: &Buffer, destination: &Buffer, from: u64, to: u64, bytes: u64) -> TransferItem<'static> {
        // SAFETY of the lifetime: each test keeps both buffers alive until
        // its batches complete.
        let (source, destination) = unsafe { (&*(source as *const Buffer), &*(destination as *const Buffer)) };
        TransferItem::Run(TransferRun { source, source_offset: from, destination, destination_offset: to, bytes })
    }

    /// The batch's runs land byte for byte where they say, and nothing else
    /// moves.
    #[test]
    fn a_batch_lands_its_runs_exactly() {
        let Some((_exclusive, crate::OpenedCuda { service: device, .. })) = crate::test_device() else { return };
        let (source, destination) = buffers(&device, |at| (at * 7 + at / 4096) as u8);
        let queue = TransferQueue::new(&device).unwrap();
        let half = BYTES / 2;
        // The second half first, then the first half, in one batch, each
        // signalling a word of its own (device memory) once landed.
        let words = device.allocate_device(4096, 4, &crate::test_entry()).unwrap();
        let signal = |offset: u64, value: u32| {
            // SAFETY of the lifetime: `words` outlives the batch.
            TransferItem::Signal { word: unsafe { &*(&words as *const Buffer) }, offset, value }
        };
        queue
            .copy(&[run(&source, &destination, half, 0, half), signal(0, 7), run(&source, &destination, 0, half, half), signal(4, 9)], 1)
            .unwrap();
        let start = Instant::now();
        while !queue.completed(1).unwrap() {
            assert!(start.elapsed() < Duration::from_secs(10), "a transfer never completed");
            std::thread::yield_now();
        }
        let mut signalled = [0u8; 8];
        words.download_at(0, &mut signalled).unwrap();
        assert_eq!(signalled, [7, 0, 0, 0, 9, 0, 0, 0], "each signal is written");
        let mut landed = vec![0u8; BYTES as usize];
        destination.download_at(0, &mut landed).unwrap();
        for (at, byte) in landed.iter().enumerate() {
            let from = (at as u64 + half) % BYTES;
            assert_eq!(*byte, (from * 7 + from / 4096) as u8, "byte {at}");
        }
        // Back to the host: the device's first 64 KiB into pinned memory,
        // after a copy within the device of its second 64 KiB over them.
        const BACK: u64 = 64 << 10;
        let returned = device.allocate_mapped(BACK, &crate::test_entry()).unwrap();
        queue.copy(&[run(&destination, &destination, BACK, 0, BACK), run(&destination, &returned, 0, 0, BACK)], 2).unwrap();
        while !queue.completed(2).unwrap() {
            assert!(start.elapsed() < Duration::from_secs(10), "a transfer never completed");
            std::thread::yield_now();
        }
        // SAFETY: within the mapping, which the completed batch wrote.
        let back = unsafe { std::slice::from_raw_parts(returned.host_address().unwrap().as_ptr(), BACK as usize) };
        assert!(back == &landed[BACK as usize..2 * BACK as usize], "the device's bytes return to the host");
        // An invalid step anywhere refuses the whole batch, queuing nothing:
        // a run between host memory, a signal past or within a word.
        for invalid in [run(&source, &returned, 0, 0, 4), signal(4094, 1), signal(2, 1)] {
            assert!(queue.copy(&[run(&source, &destination, 0, 0, 4), invalid], 3).is_err());
        }
        assert!(!queue.completed(3).unwrap(), "a refused batch queued nothing");
    }

    /// A transfer runs beside a kernel: it completes while a kernel queued
    /// before it on the execution stream still runs. It reports the rate of
    /// a transfer alone.
    #[test]
    fn a_transfer_runs_beside_a_running_kernel() {
        let Some((_exclusive, crate::OpenedCuda { service: device, device: description })) = crate::test_device() else { return };
        let (source, destination) = buffers(&device, |at| at as u8);
        let queue = TransferQueue::new(&device).unwrap();
        // The rate alone, over a warm batch and a timed one.
        queue.copy(&[run(&source, &destination, 0, 0, BYTES)], 1).unwrap();
        while !queue.completed(1).unwrap() {
            std::thread::yield_now();
        }
        let start = Instant::now();
        queue.copy(&[run(&source, &destination, 0, 0, BYTES)], 2).unwrap();
        while !queue.completed(2).unwrap() {
            std::thread::yield_now();
        }
        let rate = BYTES as f64 / start.elapsed().as_secs_f64() / 1e9;
        // The rate is the link's, which this test does not know: reported.
        eprintln!("transfer rate: {rate:.2} GB/s over {} MiB", BYTES >> 20);

        // A kernel spinning about 0.5 s, then a transfer queued behind it.
        let capability = description.device_info().compute_capability;
        let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
        let cubin = crate::nvrtc::compile_cubin_named(SPIN, "spin.cu", architecture, &["spin"]).unwrap();
        let module = DirectModule::load(&device, &cubin.image, &[cubin.lowered_names[0].as_str()]).unwrap();
        let out = device.allocate_mapped(4, &crate::test_entry()).unwrap();
        let scalars = device.allocate_mapped(4, &crate::test_entry()).unwrap();
        let clock = Instant::now();
        let anchor = TimelineAnchor::record(&device, || clock.elapsed().as_secs_f64()).unwrap();
        let buffers = [(&out, 0)];
        let words = 1_000_000_000u64.to_le_bytes();
        let mut batch = DirectBatch::new(&device, crate::NativeStream::Compute).unwrap();
        batch
            .launch(&DirectLaunch {
                module: &module,
                function: 0,
                buffers: &buffers,
                words: &words,
                scalar_results: Some((&scalars, 0)),
                grid: [1, 1, 1],
                block: [1, 1, 1],
                shared_bytes: 0,
            })
            .unwrap();
        let kernel = batch.commit().unwrap();
        queue.copy(&[run(&source, &destination, 0, 0, BYTES)], 3).unwrap();
        while !queue.completed(3).unwrap() {
            std::thread::yield_now();
        }
        let landed = clock.elapsed().as_secs_f64();
        assert!(!kernel.completed().unwrap(), "the kernel ended before the transfer landed: no overlap was observed");
        kernel.finish().unwrap();
        let (began, ended) = kernel.device_interval(&anchor).unwrap();
        eprintln!("kernel {began:.4}..{ended:.4} s, transfer landed at {landed:.4} s");
        assert!(began < landed && landed < ended, "the transfer landed outside the kernel's run");
    }

    /// Each block's threads sum a strided share of `bytes` bytes of `data`,
    /// writing the sum so nothing is elided: a read of every byte, as a
    /// resident step's weight reads are.
    const SUM: &str = r#"
struct Words { unsigned long long words; };
extern "C" __global__ void sum(const unsigned long long* data, unsigned long long* out, Words words, unsigned int* scalars) {
    unsigned long long acc = 0;
    for (unsigned long long at = blockIdx.x * blockDim.x + threadIdx.x; at < words.words; at += (unsigned long long)gridDim.x * blockDim.x)
        acc += data[at];
    if (acc == 0x5eed) out[0] = acc;
}
"#;

    /// Where the device's memory is the host's (GB10), a pool block of
    /// registered host pages, which the host addresses in place, reads as
    /// fast as one of device memory: a kernel reading 1 GiB of each, in
    /// alternating rounds, takes within 3% of the other at the median.
    /// Elsewhere it checks nothing.
    #[test]
    fn where_memory_is_the_hosts_a_host_block_reads_as_a_device_one() {
        let Some((_exclusive, opened)) = crate::test_device() else { return };
        let descriptor = crate::describe(0).unwrap();
        if !descriptor.memory_is_host() {
            return;
        }
        let device = opened.service;
        let capability = opened.device.device_info().compute_capability;
        let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
        let cubin = crate::nvrtc::compile_cubin_named(SUM, "sum.cu", architecture, &["sum"]).unwrap();
        let module = DirectModule::load(&device, &cubin.image, &[cubin.lowered_names[0].as_str()]).unwrap();
        const READ: u64 = 1 << 30;
        let block = |kind| device.allocate_block(kind, READ, &crate::test_entry(), seismic_memory::SlabFill::Zeroed).unwrap();
        let (host, own) = (block(crate::topology::PINNED_KIND), block(crate::topology::DEVICE_KIND));
        assert!(host.host_address().is_some() && own.host_address().is_none());
        let out = device.allocate_mapped(8, &crate::test_entry()).unwrap();
        let scalars = device.allocate_mapped(4, &crate::test_entry()).unwrap();
        let words = (READ / 8).to_le_bytes();
        let time = |data: &Buffer| {
            let buffers = [(data, 0), (&out, 0)];
            let mut batch = DirectBatch::new(&device, crate::NativeStream::Compute).unwrap();
            batch
                .launch(&DirectLaunch {
                    module: &module,
                    function: 0,
                    buffers: &buffers,
                    words: &words,
                    scalar_results: Some((&scalars, 0)),
                    grid: [1024, 1, 1],
                    block: [256, 1, 1],
                    shared_bytes: 0,
                })
                .unwrap();
            let submission = batch.commit().unwrap();
            submission.finish().unwrap();
            submission.device_seconds().unwrap()
        };
        let (mut on_host, mut on_device) = (Vec::new(), Vec::new());
        for _ in 0..21 {
            on_host.push(time(&host));
            on_device.push(time(&own));
        }
        let median = |times: &mut Vec<f64>| {
            times.sort_by(f64::total_cmp);
            times[times.len() / 2]
        };
        let (host, own) = (median(&mut on_host), median(&mut on_device));
        eprintln!(
            "1 GiB read: host block {:.3} ms ({:.1} GB/s), device block {:.3} ms ({:.1} GB/s)",
            host * 1e3,
            READ as f64 / host / 1e9,
            own * 1e3,
            READ as f64 / own / 1e9
        );
        assert!(host <= own * 1.03, "the host block read {host:.6} s against the device one's {own:.6} s");
    }
}
