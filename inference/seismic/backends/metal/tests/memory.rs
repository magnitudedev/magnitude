//! What Metal work leaves in the process once it is released: nothing of
//! ours. These tests have a process to themselves (an integration test
//! binary) and run one at a time, so that a reading of the process's
//! memory is a reading of the work under test.
#![cfg(target_os = "macos")]

use seismic_compiler::executable::DeviceService;
use seismic_metal::toolchain::MetalToolchain;
use seismic_metal::{DeviceHandle, DirectBatch, DirectLaunch, MetalDevice};
use seismic_memory::entry::{Entry, Tally};
use seismic_topology::StoreIdentity;
use seismic_native_target::process::{return_freed_pages, ProcessMemory};
use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};
use std::sync::Mutex;

/// One test reads the process at a time.
static ALONE: Mutex<()> = Mutex::new(());

const SOFTFLOAT: &str = include_str!("../src/softfloat.metal");

fn opened() -> (MetalDevice, MetalToolchain) {
    let device = MetalDevice::open(
        DeviceHandle::system_default().unwrap(),
        std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam)),
    ).unwrap();
    let description = seismic_metal::profile::open_device(&device, &seismic_native_target::MemoryStore::default()).unwrap();
    let toolchain = MetalToolchain::new(device.clone(), description.device_info());
    (device, toolchain)
}

/// A program of a real kernel source's size (the soft-float prelude every
/// rendered source carries) that no compile before it has seen: its
/// kernel's text carries `index`. Two instances of one templated kernel.
fn program(index: usize) -> ProgramSource {
    ProgramSource {
        text: format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{SOFTFLOAT}\n\
             template <uint TILE> kernel void residue_probe(device float* x [[buffer(0)]], \
             uint i [[thread_position_in_grid]]) {{ x[i] = x[i] * float(TILE) + float({index}); }}\n"
        ),
        entries: vec![
            ProgramEntry::named("residue_probe<2>"),
            ProgramEntry::named("residue_probe<4>"),
        ],
    }
}

fn grew(part: fn(&ProcessMemory) -> Option<u64>, from: &ProcessMemory, to: &ProcessMemory) -> i64 {
    part(to).unwrap() as i64 - part(from).unwrap() as i64
}

/// A thread that submits work for the life of the process keeps nothing of
/// a submission once it completes. Without an autorelease pool around each
/// call into Metal a submission left 180 bytes of live heap on its thread
/// (M4 Max, 2026-10-09: 20,000 submissions, +3.5 MiB); with one, none.
#[test]
fn a_long_run_of_submissions_leaves_no_live_heap() {
    let _alone = ALONE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    const SUBMISSIONS: usize = 10_000;
    /// What a submission left on a thread with no pool.
    const UNPOOLED_BYTES: i64 = 180;
    let (device, toolchain) = opened();
    let source = ProgramSource {
        text: "#include <metal_stdlib>\nusing namespace metal;\n\
               kernel void bump(device float* x [[buffer(0)]], \
               uint i [[thread_position_in_grid]]) { x[i] += 1; }\n"
            .into(),
        entries: vec![ProgramEntry::named("bump")],
    };
    let pipelines = toolchain.compile(&source).unwrap().1;
    let entry: std::sync::Arc<dyn Entry> = std::sync::Arc::new(Tally::unbounded(StoreIdentity::HostRam));
    let values = device.allocate(seismic_metal::topology::SHARED_KIND, 4096, 16, &entry).unwrap();
    let scalars = device.allocate(seismic_metal::topology::SHARED_KIND, 64, 16, &entry).unwrap();
    let submit = || {
        let mut batch = DirectBatch::observed(&device).unwrap();
        batch
            .encode(&DirectLaunch {
                pipeline: &pipelines[0],
                buffers: &[(&values, 0)],
                words: &[],
                scalar_results: Some((&scalars, 0)),
                threadgroups: [4, 1, 1],
                grid: None,
                threads_per_threadgroup: [64, 1, 1],
                threadgroup_bytes: 0,
            })
            .unwrap();
        let submission = batch.commit();
        submission.finish().unwrap();
        assert!(submission.device_seconds().unwrap() >= 0.);
    };
    // What the first submissions set up once is not what each one leaves.
    for _ in 0..SUBMISSIONS / 10 {
        submit();
    }
    let before = ProcessMemory::read();
    for _ in 0..SUBMISSIONS {
        submit();
    }
    let after = ProcessMemory::read();
    let live = grew(|reading| reading.malloc_live_bytes, &before, &after);
    println!(
        "MEASURE {SUBMISSIONS} submissions: live heap {live:+} bytes ({:+.1} a submission), \
         footprint {:+} bytes",
        live as f64 / SUBMISSIONS as f64,
        grew(|reading| reading.footprint_bytes, &before, &after),
    );
    // Measured with the pool: nothing, or one step of 0.17 MiB that does
    // not grow with the submissions (20,000 leave what 4,096 do). A thread
    // with no pool keeps twice this bound.
    assert!(
        live < SUBMISSIONS as i64 * UNPOOLED_BYTES / 2,
        "{live} bytes of live heap stay after {SUBMISSIONS} submissions"
    );
}

/// Programs formed and released leave no live heap of ours, and nothing in
/// the footprint beyond what Metal keeps for each pipeline state it has
/// formed, once the freed pages are returned. Measured before the compile
/// ran in a pool (M4 Max, 2026-10-09; 1,024 programs of the dense output
/// projection, one pipeline each): 22 KiB of live heap a program while its
/// thread lived, 37 KiB of footprint a program; with the pool, no live heap
/// and 5.4 KiB of footprint a pipeline in Metal's own `IOAccelerator`
/// region, which no release returns. That region holds each pipeline's
/// compiled code, so the small kernels here leave less (about 1 KiB).
#[test]
fn programs_formed_and_released_leave_only_what_metal_keeps_a_pipeline() {
    let _alone = ALONE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    /// Live heap a compiled program left on a thread with no pool.
    const UNPOOLED_BYTES: i64 = 22 << 10;
    /// Footprint Metal keeps for a pipeline state it formed, as measured
    /// for the engine's kernels.
    const PIPELINE_BYTES: i64 = 5530;
    let (_device, toolchain) = opened();
    // The first compiles set up what the framework holds once, and it
    // lets some of that go over the compiles that follow: what a program
    // leaves is read over the whole ladder, past that.
    for index in 0..16 {
        drop(toolchain.compile(&program(index)).unwrap());
    }
    return_freed_pages();
    let before = ProcessMemory::read();
    let mut formed = 0usize;
    const LADDER: [usize; 3] = [32, 64, 128];
    for rung in LADDER {
        for index in 0..rung {
            drop(toolchain.compile(&program(1000 * rung + index)).unwrap());
        }
        formed += rung;
        let held = ProcessMemory::read();
        return_freed_pages();
        let returned = ProcessMemory::read();
        let pipelines = 2 * formed as i64;
        let live = grew(|reading| reading.malloc_live_bytes, &before, &returned);
        let graphics = grew(|reading| reading.graphics_bytes, &before, &returned);
        let footprint = grew(|reading| reading.footprint_bytes, &before, &returned);
        println!(
            "MEASURE {formed} programs ({pipelines} pipelines) formed and released: footprint \
             {:+} bytes before the freed pages were returned and {footprint:+} after \
             ({:+.0} a program); live heap {live:+} ({:+.0} a program); graphics {graphics:+} \
             ({:+.0} a pipeline)",
            grew(|reading| reading.footprint_bytes, &before, &held),
            footprint as f64 / formed as f64,
            live as f64 / formed as f64,
            graphics as f64 / pipelines as f64,
        );
        if formed < LADDER.iter().sum() {
            continue;
        }
        // Measured with the pool: none. A thread with no pool keeps twice
        // this bound.
        assert!(
            live < formed as i64 * UNPOOLED_BYTES / 2,
            "{live} bytes of live heap stay after {formed} programs"
        );
        assert!(
            footprint <= pipelines * PIPELINE_BYTES,
            "{footprint} bytes of footprint stay after {pipelines} pipelines"
        );
    }
}

/// A pool block is one buffer Metal counts once in the device's allocated
/// size (`currentAllocatedSize`, which the device's grant is read from), at
/// exactly the bytes charged to its entry, every page of it resident in the
/// process's footprint as it returns, and a dropped block leaves both where
/// they started.
#[test]
fn a_block_is_charged_and_counted_once_and_backed_as_it_returns() {
    let _alone = ALONE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    const BLOCK: u64 = 64 << 20;
    let device = MetalDevice::open(
        DeviceHandle::system_default().unwrap(),
        std::sync::Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam)),
    ).unwrap();
    let tally = Tally::unbounded(StoreIdentity::HostRam);
    let entry: std::sync::Arc<dyn Entry> = std::sync::Arc::new(tally.clone());
    let allocated = || device.handle().current_allocated_bytes();
    let before = (allocated(), ProcessMemory::read());
    let block = device.allocate_block(seismic_metal::topology::SHARED_KIND, BLOCK, &entry, seismic_memory::SlabFill::Zeroed).unwrap();
    assert_eq!(tally.held(), BLOCK, "a block of whole pages is charged its bytes");
    assert_eq!(allocated() - before.0, BLOCK, "Metal counts the block once");
    let footprint = grew(|reading| reading.footprint_bytes, &before.1, &ProcessMemory::read());
    assert!(footprint >= BLOCK as i64 * 9 / 10, "the block's pages are backed as it returns: {footprint} bytes");
    let mut bytes = vec![1; BLOCK as usize];
    device.read(&block, 0, &mut bytes).unwrap();
    assert!(bytes.iter().all(|byte| *byte == 0), "a block reads zero");
    drop(block);
    assert_eq!(tally.held(), 0);
    assert_eq!(allocated(), before.0);
}
