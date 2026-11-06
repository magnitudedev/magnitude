//! How the read engine serves several record reads at once: the measurement
//! behind the law a streamed decode's timeline prices its reads by
//! (`seismic_io`, "How reads share the engine"). Each request reads one record of `bytes` at a random
//! offset of a real model file, uncached, each pass after its pages are
//! evicted: `k` requests made at once, and `k` one after another,
//! each waited for, at each record size; every time the fastest of
//! `REPEATS`. Against three laws: `k · (λ + b/D)` (reads served one after
//! another), `λ + k · b/D` (latencies overlapped), and `⌈k b / W⌉ · λ +
//! k · b/D` (a latency a window of `W` bytes in flight, the link's knee).
//! The file is the target of the catalog acquisition object
//! `MAGNITUDE_ARTIFACTS` names. Run alone on the device, with
//! `-p seismic-io --release --test read_law -- --ignored --nocapture`.

use seismic_io::{Buffer, CachePolicy, Range, Reader, Want};
use std::sync::Arc;
use std::time::Instant;

const REPEATS: usize = 5;

/// Where record `index` of `bytes` lies in its sinks, as staging lays
/// records: each at its own aligned offset of one destination.
fn at(index: usize, bytes: usize) -> usize {
    index * bytes.next_multiple_of(seismic_io::ALIGNMENT as usize)
}

/// A deterministic stream of offsets over the file.
struct Offsets(u64);

impl Offsets {
    fn next(&mut self, size: u64, bytes: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) % (size - bytes)
    }
}

/// Seconds to read `k` records of `bytes`: submitted at once (`together`)
/// or each waited for before the next.
fn pass(reader: &Reader, sinks: &Arc<Buffer>, offsets: &mut Offsets, k: usize, bytes: usize, together: bool) -> f64 {
    reader.evict_cached().unwrap();
    let ranges = (0..k)
        .map(|index| Range { offset: offsets.next(reader.size(), bytes as u64), length: bytes, into: sinks.clone(), at: at(index, bytes) })
        .collect::<Vec<_>>();
    let began = Instant::now();
    if together {
        // SAFETY: each sink is its own record of the allocation.
        let requests = ranges.into_iter().map(|range| unsafe { reader.request(vec![range], Want::now()) }.unwrap()).collect::<Vec<_>>();
        for request in requests {
            request.wait();
        }
    } else {
        for range in ranges {
            // SAFETY: as above.
            unsafe { reader.request(vec![range], Want::now()) }.unwrap().wait();
        }
    }
    began.elapsed().as_secs_f64()
}

#[test]
#[ignore = "times the read engine on a real model file (`MAGNITUDE_ARTIFACTS`)"]
fn the_engine_serves_record_reads_by_its_law() {
    let artifacts = std::env::var_os("MAGNITUDE_ARTIFACTS").expect("set MAGNITUDE_ARTIFACTS");
    let acquisition: serde_json::Value = serde_json::from_slice(&std::fs::read(&artifacts).unwrap()).unwrap();
    let target = acquisition["target"].as_str().expect("the acquisition object names its target");
    seismic_io::probe_volume(&seismic_io::OpenFile::open(target).unwrap(), None).unwrap();
    let reader = Reader::open(target, CachePolicy::Uncached).unwrap();
    let window = reader.knee();
    let mut offsets = Offsets(0x5eed);
    for bytes in [262_144usize, 1_900_544, 8_388_608] {
        let sinks = Arc::new(Buffer::new(at(32, bytes)));
        let fastest = |offsets: &mut Offsets, k: usize, together: bool| {
            (0..REPEATS).map(|_| pass(&reader, &sinks, offsets, k, bytes, together)).fold(f64::INFINITY, f64::min)
        };
        // One record alone, which `k · one` multiplies.
        let one = fastest(&mut offsets, 1, true);
        for k in [1usize, 2, 4, 8, 16, 32] {
            let together = fastest(&mut offsets, k, true);
            let serial = fastest(&mut offsets, k, false);
            let moved = (k * bytes) as f64;
            eprintln!(
                "record {bytes} B · k {k:2} · together {:8.3} ms ({:5.2} GB/s) · one after another {:8.3} ms ({:5.2} GB/s) · k·one {:8.3} ms · windows {}",
                together * 1e3,
                moved / together / 1e9,
                serial * 1e3,
                moved / serial / 1e9,
                k as f64 * one * 1e3,
                (k as u64 * bytes as u64).div_ceil(window),
            );
        }
    }
}
