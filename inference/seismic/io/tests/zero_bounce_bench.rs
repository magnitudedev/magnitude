//! Standalone direct-vs-bounce NVMe micro-benchmark for `seismic-io`.
//!
//! Creates a temporary multi-hundred-MiB file on the current filesystem (so it
//! sits on real NVMe storage, not `/tmp` tmpfs), evicts the page cache before
//! each pass, and compares:
//! 1. **Bounced (baseline):** unaligned 32-byte-grain GGUF-like offsets read
//!    into `Buffer::new(chunk)` at `at = 0` (~100% bounce `memcpy`).
//! 2. **Zero-bounce (`AlignedRange` + `Buffer::with_headroom`):** the exact
//!    same unaligned offsets read directly into `Buffer::with_headroom(chunk)`
//!    (0% bounce `memcpy`).
//!
//! Both modes are measured standalone and beside concurrent multi-threaded
//! DDR5 memory-copy load (simulating concurrent PCIe DMA + CPU RAM bandwidth
//! contention on 3-tier NVMe hosts).
//!
//! Run on an NVMe host with:
//! `cargo test -p seismic-io --release --test zero_bounce_bench -- --ignored --nocapture`

use seismic_io::accounting;
use seismic_io::{Buffer, CachePolicy, Ended, OpenFile, Range, Reader, Urgent, Want};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MIB: usize = 1 << 20;

/// Guard that removes the temporary benchmark file on drop (even on panic).
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn create_bench_file(size_mib: usize) -> TempFile {
    let dir = std::env::var_os("SEISMIC_IO_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current dir"));
    let path = dir.join(format!("seismic-io-zero-bounce-bench-{}.bin", std::process::id()));
    let guard = TempFile(path.clone());
    let mut file = File::create(&path).expect("create bench file");
    let block: Vec<u8> = (0..4 * MIB).map(|i| ((i * 131 + (i >> 12) * 17) & 0xff) as u8).collect();
    let mut written = 0usize;
    let total = size_mib * MIB;
    while written < total {
        let n = (total - written).min(block.len());
        file.write_all(&block[..n]).expect("write bench file");
        written += n;
    }
    file.sync_all().expect("sync bench file");
    guard
}

struct PassResult {
    wall_gb_s: f64,
    busy_gb_s: f64,
    direct_bytes: u64,
    bounced_bytes: u64,
    bounce_pct: f64,
}

fn run_pass(
    reader: &Reader,
    offsets: &[u64],
    chunk: usize,
    depth: usize,
    zero_bounce: bool,
) -> PassResult {
    reader.evict_cached().expect("evict page cache");
    let pool: Vec<Arc<Buffer>> = (0..depth)
        .map(|_| {
            Arc::new(if zero_bounce {
                Buffer::with_headroom(chunk)
            } else {
                Buffer::new(chunk)
            })
        })
        .collect();
    let volume = reader.volume();
    let before = accounting::totals(volume);
    let began = Instant::now();

    let mut flight = std::collections::VecDeque::with_capacity(depth);
    let mut next = 0usize;
    let mut checksum = 0u64;

    while next < offsets.len() || !flight.is_empty() {
        while flight.len() < depth && next < offsets.len() {
            let slot = next % depth;
            let buf = pool[slot].clone();
            let offset = offsets[next];
            let (range, lead) = if zero_bounce {
                let aligned = reader.aligned_range(offset, chunk, buf.clone(), 0);
                (aligned.range, aligned.lead)
            } else {
                (Range { offset, length: chunk, into: buf.clone(), at: 0 }, 0)
            };
            // SAFETY: each slot is only reused after its previous handle completes.
            let handle = unsafe { reader.request(vec![range], Want::streamed()) }.expect("submit");
            flight.push_back((handle, buf, lead));
            next += 1;
        }
        if let Some((handle, buf, lead)) = flight.pop_front() {
            assert!(matches!(handle.wait(), Ended::Read { failed } if failed.is_empty()));
            // Touch first and last byte to ensure data landed and isn't optimized away.
            let bytes = unsafe { buf.bytes() };
            checksum = checksum.wrapping_add(bytes[lead] as u64 + bytes[lead + chunk - 1] as u64);
        }
    }

    let elapsed = began.elapsed().as_secs_f64();
    let totals = accounting::totals(volume).since(&before);
    let payload_bytes = (offsets.len() * chunk) as f64;
    let wall_gb_s = payload_bytes / elapsed / 1e9;
    let busy_gb_s = totals.rate().unwrap_or(0.0) / 1e9;
    let bounce_pct = totals.bounce_rate().unwrap_or(0.0) * 100.0;
    std::hint::black_box(checksum);

    PassResult {
        wall_gb_s,
        busy_gb_s,
        direct_bytes: totals.direct_bytes,
        bounced_bytes: totals.bounced_bytes,
        bounce_pct,
    }
}

/// Spawns `threads` background DDR5 copy workers that continuously stream
/// 64 MiB buffers to create realistic host memory bandwidth contention.
fn start_ddr5_load(threads: usize) -> (Arc<AtomicBool>, Arc<AtomicU64>, Vec<std::thread::JoinHandle<()>>) {
    let stop = Arc::new(AtomicBool::new(false));
    let copied = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(threads);
    for t in 0..threads {
        let stop = stop.clone();
        let copied = copied.clone();
        handles.push(
            seismic_threads::spawn(format!("ddr5-load-{t}"), move || {
                const BUF_LEN: usize = 64 * MIB;
                let src = vec![(t as u8).wrapping_add(1); BUF_LEN];
                let mut dst = vec![0u8; BUF_LEN];
                while !stop.load(Ordering::Relaxed) {
                    dst.copy_from_slice(&src);
                    std::hint::black_box(dst[0]);
                    copied.fetch_add(BUF_LEN as u64, Ordering::Relaxed);
                }
            })
            .expect("spawn ddr5 load worker"),
        );
    }
    (stop, copied, handles)
}

#[test]
#[ignore = "NVMe hardware micro-benchmark comparing bounced vs zero-bounce direct reads"]
fn direct_vs_bounce_nvme_microbench() {
    let file_mib = 1024usize; // 1 GiB real NVMe file
    let chunk = 2 * MIB;
    let depth = 16usize;
    let temp = create_bench_file(file_mib);

    let open = OpenFile::open(&temp.0).expect("open bench file");
    let urgent = Urgent { parts: vec![chunk as u64], records: 2 };
    let _ = seismic_io::probe_volume(&open, Some(&urgent));
    let reader = Reader::of_urgent(&open, CachePolicy::Uncached, &urgent).expect("open reader");

    // Generate 256 unaligned 32-byte-grain GGUF-like offsets spanning the 1 GiB file (512 MiB per pass).
    let count = 256usize;
    let stride = ((file_mib * MIB) - chunk - 4096) / count;
    let offsets: Vec<u64> = (0..count)
        .map(|i| {
            // Pseudo-random permutation across the file, always misaligned mod 4096 (32-byte GGUF grain).
            let slot = (i * 73 + 19) % count;
            let base = (slot * stride) as u64;
            let misalign = ((i % 127) as u64 + 1) * 32; // never 0 mod 4096
            (base & !4095) + misalign
        })
        .collect();

    // Warm up once
    let _ = run_pass(&reader, &offsets[..32], chunk, depth, true);

    // 1) Standalone NVMe (no concurrent DDR5 load)
    let standalone_bounced = run_pass(&reader, &offsets, chunk, depth, false);
    let standalone_direct = run_pass(&reader, &offsets, chunk, depth, true);
    let standalone_delta = (standalone_direct.wall_gb_s / standalone_bounced.wall_gb_s - 1.0) * 100.0;

    // 2) Under concurrent DDR5 copy load (simulating concurrent PCIe DMA + CPU RAM traffic)
    let ddr_threads = std::thread::available_parallelism().map(|n| n.get().clamp(2, 8)).unwrap_or(4);
    let (stop, copied, workers) = start_ddr5_load(ddr_threads);
    // Let DDR5 load ramp up briefly
    std::thread::sleep(std::time::Duration::from_millis(100));
    let ddr_before = copied.load(Ordering::Relaxed);
    let ddr_t0 = Instant::now();
    let contended_bounced = run_pass(&reader, &offsets, chunk, depth, false);
    let contended_direct = run_pass(&reader, &offsets, chunk, depth, true);
    let ddr_elapsed = ddr_t0.elapsed().as_secs_f64();
    let ddr_after = copied.load(Ordering::Relaxed);
    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join().unwrap();
    }
    let ddr_bw_gb_s = ((ddr_after - ddr_before) as f64 * 2.0) / ddr_elapsed / 1e9; // read + write
    let contended_delta = (contended_direct.wall_gb_s / contended_bounced.wall_gb_s - 1.0) * 100.0;

    // 3) Verify Reader::stream on an unaligned offset achieves 0% bounce
    reader.evict_cached().expect("evict page cache");
    let before = accounting::totals(reader.volume());
    let mut streamed_bytes = 0u64;
    reader
        .stream::<std::io::Error>(32, (64 * MIB) as u64, Want::streamed(), |_, bytes| {
            streamed_bytes += bytes.len() as u64;
            std::hint::black_box(bytes[0]);
            Ok(())
        })
        .expect("stream");
    let stream_totals = accounting::totals(reader.volume()).since(&before);
    let stream_bounce_pct = stream_totals.bounce_rate().unwrap_or(0.0) * 100.0;

    eprintln!("=== seismic-io zero-bounce NVMe micro-benchmark ({count} x {} MiB = {} MiB/pass, depth={depth}) ===", chunk / MIB, count * chunk / MIB);
    eprintln!(
        "STANDALONE bounced: wall={:.2} GB/s  busy={:.2} GB/s  direct={} B  bounced={} B  bounce={:.2}%",
        standalone_bounced.wall_gb_s,
        standalone_bounced.busy_gb_s,
        standalone_bounced.direct_bytes,
        standalone_bounced.bounced_bytes,
        standalone_bounced.bounce_pct
    );
    eprintln!(
        "STANDALONE direct : wall={:.2} GB/s  busy={:.2} GB/s  direct={} B  bounced={} B  bounce={:.2}%  delta={:+.1}%",
        standalone_direct.wall_gb_s,
        standalone_direct.busy_gb_s,
        standalone_direct.direct_bytes,
        standalone_direct.bounced_bytes,
        standalone_direct.bounce_pct,
        standalone_delta
    );
    eprintln!(
        "CONTENDED ({ddr_threads} DDR5 threads ~{ddr_bw_gb_s:.1} GB/s) bounced: wall={:.2} GB/s  busy={:.2} GB/s  bounce={:.2}%",
        contended_bounced.wall_gb_s,
        contended_bounced.busy_gb_s,
        contended_bounced.bounce_pct
    );
    eprintln!(
        "CONTENDED ({ddr_threads} DDR5 threads ~{ddr_bw_gb_s:.1} GB/s) direct : wall={:.2} GB/s  busy={:.2} GB/s  bounce={:.2}%  delta={:+.1}%",
        contended_direct.wall_gb_s,
        contended_direct.busy_gb_s,
        contended_direct.bounce_pct,
        contended_delta
    );
    eprintln!(
        "STREAM (64 MiB @ offset=32): streamed={streamed_bytes} B  direct={} B  bounced={} B  bounce={:.2}%",
        stream_totals.direct_bytes, stream_totals.bounced_bytes, stream_bounce_pct
    );

    assert!(standalone_bounced.bounce_pct > 99.0, "baseline should bounce >99%, got {:.2}%", standalone_bounced.bounce_pct);
    assert!(standalone_direct.bounce_pct < 1.0, "zero-bounce should bounce <1%, got {:.2}%", standalone_direct.bounce_pct);
    assert!(contended_direct.bounce_pct < 1.0, "contended zero-bounce should bounce <1%, got {:.2}%", contended_direct.bounce_pct);
    assert!(stream_bounce_pct < 1.0, "Reader::stream should bounce <1%, got {:.2}%", stream_bounce_pct);
}
