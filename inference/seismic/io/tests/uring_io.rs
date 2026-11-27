//! Integration tests and throughput microbenchmark for the Linux `io_uring`
//! batched `O_DIRECT` read engine in `seismic-io`.

use seismic_io::accounting;
use seismic_io::{Buffer, CachePolicy, Ended, Error, OpenFile, Range, Reader, Urgent, Want};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIB: usize = 1 << 20;

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn create_test_file(name: &str, size: usize) -> (TempFile, Vec<u8>) {
    let dir = std::env::var_os("SEISMIC_IO_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let path = dir.join(format!("seismic-uring-{name}-{}.bin", std::process::id()));
    let guard = TempFile(path.clone());
    let mut data = vec![0u8; size];
    for (i, b) in data.iter_mut().enumerate() {
        *b = ((i.wrapping_mul(131).wrapping_add(i >> 11)) & 0xff) as u8;
    }
    let mut file = File::create(&path).expect("create test file");
    file.write_all(&data).expect("write test file");
    file.sync_all().expect("sync test file");
    (guard, data)
}

/// Batched QD = 32 `O_DIRECT` aligned (zero-bounce) and unaligned reads return
/// exact bytes on a single `io_uring` ring thread (on Linux) and clean up cleanly.
#[test]
fn batched_qd32_direct_and_unaligned_reads() {
    let size = 4 * MIB;
    let (temp, expected) = create_test_file("qd32", size);
    let reader = Reader::open(&temp.0, CachePolicy::Uncached).expect("open reader");

    // 32 concurrent aligned 64 KiB direct reads + 16 unaligned reads in one batch.
    let chunk = 64 * 1024;
    let mut handles = Vec::new();
    for i in 0..32usize {
        let offset = (i * chunk) as u64;
        let buf = Arc::new(Buffer::new(chunk));
        let range = Range { offset, length: chunk, into: buf.clone(), at: 0 };
        // SAFETY: each buffer is exclusively owned by its handle until `wait()`.
        let handle = unsafe { reader.request(vec![range], Want::now()) }.expect("submit aligned");
        handles.push((handle, buf, offset as usize, 0usize, chunk));
    }
    for i in 0..16usize {
        let offset = (i * chunk + 37) as u64; // deliberately unaligned
        let len = 12_345;
        let buf = Arc::new(Buffer::with_headroom(len));
        let aligned = reader.aligned_range(offset, len, buf.clone(), 0);
        // SAFETY: each buffer is exclusively owned by its handle until `wait()`.
        let handle = unsafe { reader.request(vec![aligned.range], Want::expected(1, 0.9)) }.expect("submit unaligned");
        handles.push((handle, buf, offset as usize, aligned.lead, len));
    }

    for (handle, buf, offset, lead, len) in handles {
        match handle.wait() {
            Ended::Read { failed } => assert!(failed.is_empty(), "unexpected failures: {failed:?}"),
            Ended::Panicked(p) => p.resume(),
        }
        // SAFETY: request ended.
        let got = unsafe { buf.bytes() };
        assert_eq!(&got[lead..lead + len], &expected[offset..offset + len]);
    }

    #[cfg(target_os = "linux")]
    if seismic_io::io_uring_active() {
        assert_eq!(
            seismic_io::read_workers(),
            1,
            "io_uring engine must sustain QD=32 on a single ring thread"
        );
    }
}

/// Scatter-gather multi-part aligned direct reads (`IORING_OP_READV`) land
/// directly into disjoint destinations with zero corruption.
#[test]
fn vectored_direct_scatter_gather_reads() {
    let size = 1 * MIB;
    let (temp, expected) = create_test_file("vectored", size);
    let reader = Reader::open(&temp.0, CachePolicy::Uncached).expect("open reader");

    let part_len = 16 * 1024;
    let bufs: Vec<Arc<Buffer>> = (0..4).map(|_| Arc::new(Buffer::new(part_len))).collect();
    let ranges: Vec<Range> = bufs
        .iter()
        .enumerate()
        .map(|(i, buf)| Range {
            offset: (i * part_len) as u64,
            length: part_len,
            into: buf.clone(),
            at: 0,
        })
        .collect();
    // SAFETY: disjoint buffers held until `wait()`.
    let handle = unsafe { reader.request(ranges, Want::now()) }.expect("submit vectored");
    match handle.wait() {
        Ended::Read { failed } => assert!(failed.is_empty(), "unexpected failures: {failed:?}"),
        Ended::Panicked(p) => p.resume(),
    }
    for (i, buf) in bufs.iter().enumerate() {
        // SAFETY: request ended.
        let got = unsafe { buf.bytes() };
        assert_eq!(got, &expected[i * part_len..(i + 1) * part_len]);
    }
}

/// Dropping speculative `Want::expected` handles while a burst is queued and in
/// flight cancels them cleanly (ending with `Error::Cancelled` or completing
/// safely without memory corruption) while concurrent demand reads succeed.
#[test]
fn cancel_queued_and_inflight_speculative_reads_beside_demand() {
    let size = 4 * MIB;
    let (temp, expected) = create_test_file("cancel", size);
    let reader = Reader::open(&temp.0, CachePolicy::Uncached).expect("open reader");

    let chunk = 64 * 1024;
    let mut spec_endings = Vec::new();
    let mut spec_handles = Vec::new();
    for i in 0..32usize {
        let buf = Arc::new(Buffer::new(chunk));
        let range = Range { offset: (i * chunk) as u64, length: chunk, into: buf, at: 0 };
        // SAFETY: buffer owned by request until ending finishes.
        let handle = unsafe { reader.request(vec![range], Want::expected(2, 0.5)) }.expect("submit ahead");
        spec_endings.push(handle.ending());
        spec_handles.push(handle);
    }

    // Submit a demand read alongside the speculative burst.
    let demand_buf = Arc::new(Buffer::new(chunk));
    let demand_range = Range { offset: 0, length: chunk, into: demand_buf.clone(), at: 0 };
    // SAFETY: buffer owned by request until `wait()`.
    let demand_handle = unsafe { reader.request(vec![demand_range], Want::now()) }.expect("submit demand");

    // Immediately drop all speculative handles to exercise both queue cancellation
    // and Linux `IORING_OP_ASYNC_CANCEL` on in-flight ops.
    drop(spec_handles);

    // Demand read must complete cleanly with exact bytes.
    match demand_handle.wait() {
        Ended::Read { failed } => assert!(failed.is_empty(), "demand read failed: {failed:?}"),
        Ended::Panicked(p) => p.resume(),
    }
    // SAFETY: demand request ended.
    assert_eq!(unsafe { demand_buf.bytes() }, &expected[..chunk]);

    // Every cancelled speculative request must terminate promptly without hanging.
    let deadline = Instant::now() + Duration::from_secs(10);
    for ending in spec_endings {
        let ended = ending.wait_until(deadline).expect("cancelled speculative request must end");
        match ended {
            Ended::Read { failed } => {
                assert!(
                    failed.is_empty() || failed.iter().all(|(_, e)| matches!(**e, Error::Cancelled)),
                    "unexpected non-cancel error: {failed:?}"
                );
            }
            Ended::Panicked(p) => p.resume(),
        }
    }
}

/// Throughput microbenchmark comparing `io_uring` vs `pread` worker pool at
/// QD = 16 and QD = 32 on 1 MiB and 2 MiB `O_DIRECT` reads.
///
/// Run twice on a Linux NVMe host:
///   1) `cargo test -p seismic-io --release --test uring_io -- --ignored --nocapture`
///   2) `SEISMIC_IO_DISABLE_URING=1 cargo test -p seismic-io --release --test uring_io -- --ignored --nocapture`
#[test]
#[ignore = "NVMe throughput microbenchmark for io_uring vs pread pool"]
fn uring_vs_pread_qd16_qd32_microbench() {
    let file_mib = 512usize;
    let dir = std::env::var_os("SEISMIC_IO_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current dir"));
    let path = dir.join(format!("seismic-uring-bench-{}.bin", std::process::id()));
    let _guard = TempFile(path.clone());

    {
        let mut file = File::create(&path).expect("create bench file");
        let block: Vec<u8> = (0..4 * MIB).map(|i| ((i * 131 + (i >> 12) * 17) & 0xff) as u8).collect();
        for _ in 0..(file_mib / 4) {
            file.write_all(&block).expect("write bench block");
        }
        file.sync_all().expect("sync bench file");
    }

    let open = OpenFile::open(&path).expect("open bench file");
    let backend = if seismic_io::io_uring_active() { "io_uring" } else { "pread_pool" };

    for &(chunk_mib, depth) in &[(1usize, 16usize), (2usize, 16usize), (2usize, 32usize)] {
        let chunk = chunk_mib * MIB;
        let urgent = Urgent { parts: vec![chunk as u64], records: 2 };
        let reader = Reader::of_urgent(&open, CachePolicy::Uncached, &urgent).expect("open reader");
        let count = 256usize;
        let max_slots = (file_mib * MIB) / chunk;
        let offsets: Vec<u64> = (0..count)
            .map(|i| (((i * 73 + 19) % max_slots) * chunk) as u64)
            .collect();

        let run = |offsets: &[u64]| -> (f64, f64, usize) {
            reader.evict_cached().expect("evict");
            let pool: Vec<Arc<Buffer>> = (0..depth).map(|_| Arc::new(Buffer::new(chunk))).collect();
            let before = accounting::totals(reader.volume());
            let t0 = Instant::now();
            let mut flight = std::collections::VecDeque::with_capacity(depth);
            let mut next = 0usize;
            let mut checksum = 0u64;
            while next < offsets.len() || !flight.is_empty() {
                while flight.len() < depth && next < offsets.len() {
                    let buf = pool[next % depth].clone();
                    let range = Range { offset: offsets[next], length: chunk, into: buf.clone(), at: 0 };
                    // SAFETY: slot reused only after its prior handle completes.
                    let handle = unsafe { reader.request(vec![range], Want::streamed()) }.expect("submit");
                    flight.push_back((handle, buf));
                    next += 1;
                }
                if let Some((handle, buf)) = flight.pop_front() {
                    assert!(matches!(handle.wait(), Ended::Read { failed } if failed.is_empty()));
                    let bytes = unsafe { buf.bytes() };
                    checksum = checksum.wrapping_add(bytes[0] as u64 + bytes[chunk - 1] as u64);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let totals = accounting::totals(reader.volume()).since(&before);
            let wall_gb_s = (offsets.len() * chunk) as f64 / elapsed / 1e9;
            let busy_gb_s = totals.rate().unwrap_or(0.0) / 1e9;
            std::hint::black_box(checksum);
            (wall_gb_s, busy_gb_s, seismic_io::read_workers())
        };

        // Warmup
        let _ = run(&offsets[..32]);
        let (wall_gb_s, busy_gb_s, workers) = run(&offsets);
        eprintln!(
            "[{backend}] chunk={chunk_mib} MiB  QD={depth:2}  wall={wall_gb_s:.2} GB/s  busy={busy_gb_s:.2} GB/s  workers={workers}"
        );
    }
}
