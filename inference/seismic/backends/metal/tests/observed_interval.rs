//! An observed batch's device interval, from GPU timestamp samples, lies
//! inside its submission envelope whatever its host gates did before or
//! after its commit, and what its samples cost. The streamed prefill of a
//! whole-model step on an M4 Pro (macOS 15.7.9, 2026-11-17) failed the
//! envelope when it read `GPUStartTime`: the second of four command
//! buffers, behind a host gate, reported a start 0.2 s before the batch's
//! first commit.
#![cfg(target_os = "macos")]
#![allow(
    clippy::disallowed_methods,
    reason = "a test starts threads of its own"
)]

use seismic_compiler::executable::DeviceService;
use seismic_metal::toolchain::MetalToolchain;
use seismic_metal::{host_seconds, DeviceHandle, DirectBatch, DirectLaunch, MetalDevice, SharedEvent};
use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};
use std::time::Duration;

/// When a batch's gates open, against its first commit.
#[derive(Clone, Copy, Debug)]
enum Gates {
    /// No gate: one command buffer.
    None,
    /// Every gate open before the first commit.
    OpenBefore,
    /// Each gate opened by another thread this long after the one before,
    /// the first this long after the commit.
    OpenAfter(Duration),
    /// Each gate open before the first commit, and the launch behind each
    /// parked on a word the host writes this long after the commit, as a
    /// streamed step's landing launch parks on its landing list.
    Parked(Duration),
}

fn opened() -> (MetalDevice, MetalToolchain) {
    let device = MetalDevice::open(
        DeviceHandle::system_default().unwrap(),
        std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam)),
    ).unwrap();
    let description = seismic_metal::profile::open_device(&device, &seismic_native_target::MemoryStore::default()).unwrap();
    let toolchain = MetalToolchain::new(device.clone(), description.device_info());
    (device, toolchain)
}

#[test]
fn an_observed_batch_lies_inside_its_envelope_whatever_its_gates_did() {
    let (device, toolchain) = opened();
    // About a millisecond of device work a launch.
    let source = ProgramSource {
        text: "#include <metal_stdlib>\n#pragma METAL internals : enable\nusing namespace metal;\n\
               kernel void spin(device float* x [[buffer(0)]], \
               uint i [[thread_position_in_grid]]) { \
               float v = x[i]; for (uint k = 0; k < 100000; ++k) { v = v * 0.999999f + 1e-6f; } x[i] = v; }\n\
               kernel void park(volatile coherent(system) device const int* word [[buffer(0)]], \
               uint i [[thread_position_in_grid]]) { \
               for (uint spin = 0; spin < (1u << 28) && word[0] == 0; ++spin) { \
               atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3)); } }\n"
            .into(),
        entries: vec![ProgramEntry::named("spin"), ProgramEntry::named("park")],
    };
    let pipelines = toolchain.compile(&source).unwrap().1;
    let entry: std::sync::Arc<dyn seismic_memory::entry::Entry> = std::sync::Arc::new(
        seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam),
    );
    let values = device.allocate(seismic_metal::topology::SHARED_KIND, 4096, 16, &entry).unwrap();
    let launch = DirectLaunch {
        pipeline: &pipelines[0],
        buffers: &[(&values, 0)],
        words: &[],
        scalar_results: None,
        threadgroups: [16, 1, 1],
        grid: None,
        threads_per_threadgroup: [64, 1, 1],
        threadgroup_bytes: 0,
    };
    let words = device.allocate(seismic_metal::topology::SHARED_KIND, 16, 16, &entry).unwrap();
    let park = DirectLaunch {
        pipeline: &pipelines[1],
        buffers: &[(&words, 0)],
        words: &[],
        scalar_results: None,
        threadgroups: [1, 1, 1],
        grid: None,
        threads_per_threadgroup: [32, 1, 1],
        threadgroup_bytes: 0,
    };
    const GATES: u64 = 3;
    let idle = Duration::from_millis(200);
    let mut failures = Vec::new();
    for round in 0..4 {
        for gates in [Gates::None, Gates::OpenBefore, Gates::OpenAfter(idle), Gates::Parked(Duration::from_millis(1500))] {
            let gate = SharedEvent::new(&device).unwrap();
            if let Gates::OpenBefore | Gates::Parked(_) = gates {
                gate.signal(GATES);
            }
            device.write(&words, 0, &0u32.to_le_bytes()).unwrap();
            // The device idles before the batch, as a step's does behind
            // its host.
            std::thread::sleep(idle);
            let mut batch = DirectBatch::observed(&device).unwrap();
            batch.group(1);
            batch.encode(&launch).unwrap();
            if !matches!(gates, Gates::None) {
                for value in 1..=GATES {
                    batch.wait_event(&gate, value).unwrap();
                    if let Gates::Parked(_) = gates {
                        batch.group(1);
                        batch.encode(&park).unwrap();
                    }
                    batch.group(1);
                    batch.encode(&launch).unwrap();
                }
            }
            let before = host_seconds();
            let submission = batch.commit();
            if let Gates::Parked(delay) = gates {
                std::thread::sleep(delay);
                device.write(&words, 0, &1u32.to_le_bytes()).unwrap();
            }
            if let Gates::OpenAfter(delay) = gates {
                let gate = &gate;
                std::thread::scope(|scope| {
                    scope.spawn(move || {
                        for value in 1..=GATES {
                            std::thread::sleep(delay);
                            gate.signal(value);
                        }
                    });
                });
            }
            let outcome = submission.device_interval();
            let after = host_seconds();
            println!("MEASURE round {round} {gates:?}: committed by {before:.6}, ended by {after:.6}: {outcome:?}");
            // The span holds what the device waited on between its first
            // launch and its last.
            let waited = match gates {
                Gates::None | Gates::OpenBefore => Duration::ZERO,
                Gates::OpenAfter(delay) => delay * GATES as u32,
                Gates::Parked(delay) => delay,
            };
            match outcome {
                Ok((start, end)) if end - start < 0.9 * waited.as_secs_f64() => failures.push(format!(
                    "round {round} {gates:?}: a {:.6} s span misses the {waited:?} the device waited",
                    end - start
                )),
                Ok(_) => {}
                Err(error) => failures.push(format!("round {round} {gates:?}: {error}")),
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// What an observed batch's samples add: each of its encoders samples its
/// end (and each command buffer's first its start), which a production
/// batch's do not. A batch of `ENCODERS` encoders of two small independent
/// launches each, production and observed alternately; the difference of
/// their median times from commit to completion, an encoder's share.
#[test]
#[ignore = "device micro-benchmark; run explicitly with --ignored --nocapture"]
fn an_observed_batch_prices_its_samples_per_encoder() {
    const ENCODERS: usize = 1000;
    const REPEATS: usize = 31;
    let (device, toolchain) = opened();
    let source = ProgramSource {
        text: "#include <metal_stdlib>\nusing namespace metal;\n\
               kernel void touch(device float* x [[buffer(0)]], \
               uint i [[thread_position_in_grid]]) { x[i] = x[i] * 0.5f + 1.0f; }\n"
            .into(),
        entries: vec![ProgramEntry::named("touch")],
    };
    let pipelines = toolchain.compile(&source).unwrap().1;
    let entry: std::sync::Arc<dyn seismic_memory::entry::Entry> = std::sync::Arc::new(
        seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam),
    );
    for floats in [4096u64, 1 << 18] {
        let halves = [0, 1].map(|_| device.allocate(seismic_metal::topology::SHARED_KIND, 4 * floats, 16, &entry).unwrap());
        let bound = halves.each_ref().map(|half| [(&**half, 0u64)]);
        let launches = bound.each_ref().map(|buffers| DirectLaunch {
            pipeline: &pipelines[0],
            buffers,
            words: &[],
            scalar_results: None,
            threadgroups: [floats / 256, 1, 1],
            grid: None,
            threads_per_threadgroup: [256, 1, 1],
            threadgroup_bytes: 0,
        });
        let seconds = |observed: bool| {
            let mut batch = if observed { DirectBatch::observed(&device) } else { DirectBatch::new(&device) }.unwrap();
            for _ in 0..ENCODERS {
                batch.group(2);
                for launch in &launches {
                    batch.encode(launch).unwrap();
                }
            }
            let start = host_seconds();
            let submission = batch.commit();
            submission.finish().unwrap();
            host_seconds() - start
        };
        let mut production = Vec::with_capacity(REPEATS);
        let mut observed = Vec::with_capacity(REPEATS);
        seconds(false);
        seconds(true);
        for _ in 0..REPEATS {
            production.push(seconds(false));
            observed.push(seconds(true));
        }
        let median = |samples: &mut Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[REPEATS / 2]
        };
        let (production, observed) = (median(&mut production), median(&mut observed));
        let encoder = production / ENCODERS as f64;
        let added = (observed - production) / ENCODERS as f64;
        println!(
            "MEASURE {} KiB launches, {ENCODERS} encoders of 2: production {:.3} ms ({:.3} µs an encoder), \
             observed {:.3} ms: samples add {:.3} µs an encoder, {:.1}% of its time",
            (4 * floats) >> 10,
            production * 1e3,
            encoder * 1e6,
            observed * 1e3,
            added * 1e6,
            100.0 * added / encoder
        );
    }
}
