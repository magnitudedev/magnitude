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

use crate::buffer::{copy_pieces, Buffer};
use crate::direct::HostWord;
use crate::driver::Stream;
use crate::executor::Device;
use seismic_compiler::errors::ExecutionError;
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
    /// Why the queue can complete no batch from now: a driver call failed
    /// after a batch began to be queued (the stream then holds part of it).
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
        let checked = items.iter().map(Self::check).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect::<Vec<_>>();
        let faulted = {
            let mut fault = self.fault.lock().expect("transfer fault lock is never poisoned");
            if let Some(fault) = fault.as_ref() {
                return Err(ExecutionError::SubmissionFailed(format!("the transfer queue faulted: {fault}")));
            }
            match self.queue(&checked, value).map_err(submission).and_then(|()| self.done.wake_after(&self.stream)) {
                Ok(()) => false,
                Err(error) => {
                    *fault = Some(error.to_string());
                    true
                }
            }
        };
        // Its waiters learn the fault now ([`Self::completed`]), the fault's
        // lock released first: a waiter reads it under the word's.
        if faulted {
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
        self.fault.lock().expect("transfer fault lock is never poisoned").is_some()
    }

    /// A batch's item as the driver calls that queue it: a signal, or a
    /// run's copies ([`copy_pieces`]: no copy crosses a host reservation's
    /// granule).
    fn check(item: &TransferItem<'_>) -> Result<Vec<Checked>, ExecutionError> {
        let refused = |why: &str| ExecutionError::SubmissionFailed(why.into());
        let run = match item {
            TransferItem::Run(run) => run,
            TransferItem::Signal { word, offset, value } => {
                if offset % 4 != 0 || offset.checked_add(4).is_none_or(|end| end > word.len()) {
                    return Err(refused("a transfer signal's word lies outside its buffer or is unaligned"));
                }
                return Ok(vec![Checked::Signal { word: word.pointer() + offset, value: *value }]);
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
        copy_pieces(run.source, run.source_offset, run.destination, run.destination_offset, run.bytes)
            .into_iter()
            .map(|(source_offset, destination_offset, bytes)| {
                let bytes = host(bytes)?;
                let (source, destination) = (run.source.pointer() + source_offset, run.destination.pointer() + destination_offset);
                Ok(match (source_host, destination_host) {
                    // SAFETY: within the source's mapped bytes, checked above.
                    (Some(mapped), _) => Checked::Upload { source: unsafe { mapped.as_ptr().add(host(source_offset)?) }, destination, bytes },
                    // SAFETY: within the destination's mapped bytes, checked above.
                    (None, Some(mapped)) => {
                        Checked::Download { source, destination: unsafe { mapped.as_ptr().add(host(destination_offset)?) }, bytes }
                    }
                    (None, None) => Checked::Within { source, destination, bytes },
                })
            })
            .collect()
    }

    fn queue(&self, steps: &[Checked], value: u32) -> Result<(), crate::driver::DriverError> {
        let context = self.device.context();
        let _current = context.enter()?;
        let issue = context.issuing()?;
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
        let fault = self.fault.lock().expect("transfer fault lock is never poisoned").clone();
        if fault.is_none() && reached() {
            return Ok(true);
        }
        if !self.stream.query().map_err(submission)? {
            return Ok(false);
        }
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
    use crate::RangeMemory;
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

    /// Pinned host bytes holding `pattern(i)` at byte `i`, and a reserved
    /// device range as large, as an expert bank's records are.
    fn buffers(device: &Device, pattern: impl Fn(u64) -> u8) -> (Buffer, Buffer) {
        let source = device.allocate_mapped(BYTES).unwrap();
        let host = source.host_address().unwrap();
        for at in 0..BYTES {
            // SAFETY: within the mapping, which no device work reaches yet.
            unsafe { host.as_ptr().add(at as usize).write(pattern(at)) };
        }
        let granule = crate::reservation_granule(device.ordinal(), RangeMemory::Device).unwrap().unwrap();
        let reserved = BYTES.div_ceil(granule) * granule;
        let (destination, _resizer) = device.allocate_reserved(reserved, reserved, granule, RangeMemory::Device, std::sync::Arc::new(seismic_compiler::executable::Unbounded)).unwrap();
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
        let words = device.allocate_within(4096, 4, None).unwrap();
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
        let returned = device.allocate_mapped(BACK).unwrap();
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
        let capability = description.facts().compute_capability;
        let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
        let cubin = crate::nvrtc::compile_cubin_named(SPIN, "spin.cu", architecture, &["spin"]).unwrap();
        let module = DirectModule::load(&device, &cubin.image, &[cubin.lowered_names[0].as_str()]).unwrap();
        let out = device.allocate_mapped(4).unwrap();
        let scalars = device.allocate_mapped(4).unwrap();
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

    /// Where the device's memory is the host's (GB10), device storage
    /// reserved in pinned host memory, which the host addresses in place,
    /// reads as fast as one reserved as device memory: a kernel reading
    /// 1 GiB of each, in alternating rounds, takes within 3% of the other
    /// at the median. Elsewhere it checks nothing.
    #[test]
    fn where_memory_is_the_hosts_a_host_reservation_reads_as_a_device_one() {
        let Some((_exclusive, opened)) = crate::test_device() else { return };
        let descriptor = crate::describe(0).unwrap();
        if !descriptor.memory_is_host() {
            return;
        }
        let device = opened.service;
        let capability = opened.device.facts().compute_capability;
        let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
        let cubin = crate::nvrtc::compile_cubin_named(SUM, "sum.cu", architecture, &["sum"]).unwrap();
        let module = DirectModule::load(&device, &cubin.image, &[cubin.lowered_names[0].as_str()]).unwrap();
        const READ: u64 = 1 << 30;
        let reserved = |memory| {
            let granule = crate::reservation_granule(device.ordinal(), memory).unwrap().unwrap();
            let bytes = READ.div_ceil(granule) * granule;
            device.allocate_reserved(bytes, bytes, granule, memory, std::sync::Arc::new(seismic_compiler::executable::Unbounded)).unwrap()
        };
        let (host, _host_resizer) = reserved(RangeMemory::HostNode);
        let (own, _own_resizer) = reserved(RangeMemory::Device);
        assert!(host.host_address().is_some() && own.host_address().is_none());
        let out = device.allocate_mapped(8).unwrap();
        let scalars = device.allocate_mapped(4).unwrap();
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
            "1 GiB read: host reservation {:.3} ms ({:.1} GB/s), device reservation {:.3} ms ({:.1} GB/s)",
            host * 1e3,
            READ as f64 / host / 1e9,
            own * 1e3,
            READ as f64 / own / 1e9
        );
        assert!(host <= own * 1.03, "the host reservation read {host:.6} s against the device one's {own:.6} s");
    }

    /// A registered host tier's growth reads zero in every byte it makes
    /// visible, zeroing only those its backing did not give zeroed: the
    /// bytes a shrink left in the granule it kept are zeroed, a newly backed
    /// granule is the host's, zeroed (`ReservedMemory::commit` says where they
    /// start), and the bytes the tier still holds keep theirs.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_registered_host_tiers_growth_reads_zero() {
        let Some((_exclusive, crate::OpenedCuda { service: device, .. })) = crate::test_device() else { return };
        let Some(granule) = crate::reservation_granule(device.ordinal(), RangeMemory::Registered).unwrap() else {
            eprintln!("this device reserves no registered host pages");
            return;
        };
        let size = granule as usize;
        let (tier, resizer) =
            device.allocate_reserved(granule, 3 * granule, granule, RangeMemory::Registered, std::sync::Arc::new(seismic_compiler::executable::Unbounded)).unwrap();
        let base = tier.host_address().unwrap().as_ptr();
        // SAFETY: the committed granule, which no device work reaches.
        unsafe { std::ptr::write_bytes(base, 7, size) };
        let grow = |resizer: crate::CudaResizer, bytes: u64| match resizer.prepare(bytes) {
            Ok(prepared) => prepared.install().release(),
            Err((_, error)) => panic!("a {bytes}-byte commitment: {error}"),
        };
        let resizer = grow(grow(resizer, granule / 2), 2 * granule);
        // SAFETY: two backed granules, which no device work reaches.
        let bytes = unsafe { std::slice::from_raw_parts(base, 2 * size) };
        assert!(bytes[..size / 2].iter().all(|&byte| byte == 7), "the bytes the tier held keep theirs");
        assert!(bytes[size / 2..].iter().all(|&byte| byte == 0), "the growth reads zero");
        assert_eq!(tier.reservation().unwrap().commit(3 * size).unwrap(), 2 * size, "only the third granule is newly backed");
        drop(resizer);
    }

    /// A host tier over a link (registered host pages) grows, shrinks and
    /// regrows granule by granule; its bytes read in place and copy to the
    /// device across granules (one copy per granule, `copy_pieces`); a
    /// growth whose registration the driver refuses (a granule already
    /// registered elsewhere) leaves the tier as it was; and every backed
    /// granule is counted large or small.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_registered_host_tier_grows_shrinks_and_rolls_back() {
        let Some((_exclusive, crate::OpenedCuda { service: device, .. })) = crate::test_device() else { return };
        let Some(granule) = crate::reservation_granule(device.ordinal(), RangeMemory::Registered).unwrap() else {
            eprintln!("this device reserves no registered host pages");
            return;
        };
        let before = seismic_reservation::large_page_backing();
        let (tier, resizer) =
            device.allocate_reserved(granule, 4 * granule, granule, RangeMemory::Registered, std::sync::Arc::new(seismic_compiler::executable::Unbounded)).unwrap();
        let base = tier.host_address().unwrap().as_ptr();
        let backed = |granules: u64| {
            let now = seismic_reservation::large_page_backing();
            assert_eq!(
                (now.large_bytes - before.large_bytes) + (now.small_bytes - before.small_bytes),
                granules * granule,
                "each backed granule counted once"
            );
        };
        let resize = |resizer: crate::CudaResizer, bytes: u64| match resizer.prepare(bytes) {
            Ok(prepared) => prepared.install().release(),
            Err((_, error)) => panic!("a {bytes}-byte commitment: {error}"),
        };
        let resizer = resize(resizer, 3 * granule);
        backed(3);
        let bytes = 3 * granule as usize;
        for at in 0..bytes {
            // SAFETY: within the three backed granules, which no device work reaches.
            unsafe { base.add(at).write((at % 253) as u8) };
        }
        // A copy over all three granules, from a mid-granule offset.
        let landed = device.allocate_within(bytes as u64, 16, None).unwrap();
        let queue = TransferQueue::new(&device).unwrap();
        let from = granule / 2 + 16;
        queue.copy(&[run(&tier, &landed, from, 0, 2 * granule)], 1).unwrap();
        while !queue.completed(1).unwrap() {
            std::thread::yield_now();
        }
        let mut read = vec![0u8; 2 * granule as usize];
        landed.download_at(0, &mut read).unwrap();
        assert!(read.iter().enumerate().all(|(at, &byte)| byte == ((from as usize + at) % 253) as u8), "the copy across granules lands");
        // Shrink to one granule, then refuse a regrowth: granule 1 is
        // registered by someone else first.
        let resizer = resize(resizer, granule);
        // The device is quiet: the shrink unregistered its granules at once.
        assert_eq!(tier.reservation().unwrap().registered_granules(), Some(1));
        backed(1);
        let second = unsafe { base.add(granule as usize) };
        let driver = &device.context().driver;
        let _current = device.context().enter().unwrap();
        unsafe {
            assert_eq!(libc::mprotect(second.cast(), granule as usize, libc::PROT_READ | libc::PROT_WRITE), 0);
            second.write(1);
            driver.check((driver.host_register)(second.cast(), granule as usize, 1), "foreign registration").unwrap();
        }
        let Err((resizer, _)) = resizer.prepare(3 * granule) else {
            panic!("a refused registration refuses the growth");
        };
        backed(1);
        // SAFETY: granule 0 is still backed.
        assert_eq!(unsafe { *base.add(17) }, 17, "the kept granule keeps its bytes");
        unsafe { driver.check((driver.host_unregister)(second.cast()), "foreign unregistration").unwrap() };
        let resizer = resize(resizer, 3 * granule);
        backed(3);
        // SAFETY: a regrown granule reads zero.
        assert_eq!(unsafe { *second }, 0, "a regrown granule is fresh");
        drop(resizer);
        drop(tier);
        assert_eq!(seismic_reservation::large_page_backing(), before, "a dropped tier is uncounted");
    }
}
