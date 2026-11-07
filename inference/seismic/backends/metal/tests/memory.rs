//! What Metal work leaves in the process once it is released: nothing of
//! ours. These tests have a process to themselves (an integration test
//! binary) and run one at a time, so that a reading of the process's
//! memory is a reading of the work under test.
#![cfg(target_os = "macos")]

use seismic_compiler::executable::DeviceService;
use seismic_metal::toolchain::MetalToolchain;
use seismic_reservation::Forming;
use seismic_metal::{DeviceHandle, DirectBatch, DirectLaunch, MetalDevice, MetalResizer, ViewForm};
use seismic_native_target::process::{return_freed_pages, ProcessMemory};
use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};
use std::sync::Mutex;

/// One test reads the process at a time.
static ALONE: Mutex<()> = Mutex::new(());

const SOFTFLOAT: &str = include_str!("../src/softfloat.metal");

fn opened() -> (MetalDevice, MetalToolchain) {
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let description = seismic_metal::profile::open_device(&device).unwrap();
    let toolchain = MetalToolchain::new(device.clone(), description.facts());
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
    let values = device.allocate(4096, 16).unwrap();
    let scalars = device.allocate(64, 16).unwrap();
    let submit = || {
        let mut batch = DirectBatch::observed(&device).unwrap();
        batch
            .encode(&DirectLaunch {
                pipeline: &pipelines[0],
                buffers: &[(seismic_metal::Binding::Buffer(&values), 0)],
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

/// Metal counts a reservation's backed bytes once in the device's
/// allocated size (`currentAllocatedSize`, which the device's grant is read
/// from), as Seismic charges them, in each form: chunks, or one buffer over
/// the range while every granule is backed in the whole form. A change of
/// form beside the view holds the next form's buffers beside it from its
/// prepare to its release, the lesser form's bytes more, which its holder
/// is charged for that window; a change in place holds nothing beside it.
#[test]
fn a_reservation_counts_its_backed_bytes_once_in_the_devices_allocated_size() {
    let _alone = ALONE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    const GRANULE: u64 = seismic_metal::device::RESERVATION_GRANULE as u64;
    const GRANULES: u64 = 64;
    const ALL: u64 = GRANULES * GRANULE;
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let allocated = || device.handle().current_allocated_bytes();
    let prepare_in = |resizer: MetalResizer, bytes: u64, form: ViewForm, forming: Forming| {
        resizer.prepare_as(bytes, form, forming).map_err(|(_, error)| error).unwrap()
    };
    let prepare = |resizer: MetalResizer, bytes: u64, form: ViewForm| prepare_in(resizer, bytes, form, Forming::Beside);
    let resize = |resizer: MetalResizer, bytes: u64, form: ViewForm| prepare(resizer, bytes, form).install().release();
    // An empty reservation holds its record table alone, which a reservation
    // of the same range always does.
    let before = allocated();
    let (buffer, resizer) = device.allocate_reserved(0, ALL, GRANULE, GRANULE).unwrap();
    let table = allocated() - before;
    let backed = || allocated() - before - table;
    let resizer = resize(resizer, ALL / 2, ViewForm::Chunks);
    assert_eq!(backed(), ALL / 2, "half the granules, as chunks");
    let resizer = resize(resizer, ALL, ViewForm::Chunks);
    assert!(!buffer.is_whole());
    assert_eq!(backed(), ALL, "every granule, as chunks");
    let resizer = resize(resizer, ALL - GRANULE, ViewForm::Chunks);
    let prepared = prepare(resizer, ALL, ViewForm::Whole);
    assert_eq!(backed(), (ALL - GRANULE) + ALL, "a growth into the whole form, beside the chunks");
    let resizer = prepared.install().release();
    assert!(buffer.is_whole());
    assert_eq!(backed(), ALL, "every granule, as one buffer");
    let prepared = prepare(resizer, ALL - GRANULE, ViewForm::Chunks);
    assert_eq!(backed(), ALL + (ALL - GRANULE), "a shrink from the whole form, beside it");
    let resizer = prepared.install().release();
    assert_eq!(backed(), ALL - GRANULE, "a shrink from every granule, as chunks");
    let resizer = resize(resizer, ALL, ViewForm::Whole);
    let prepared = prepare_in(resizer, ALL / 4, ViewForm::Chunks, Forming::InPlace);
    assert_eq!(backed(), ALL, "a shrink in place forms nothing beside the one buffer");
    let pending = prepared.install();
    assert_eq!(backed(), ALL / 4, "its install forms the kept chunks in the one buffer's place");
    let resizer = pending.release();
    assert_eq!(backed(), ALL / 4);
    drop((buffer, resizer));
    assert_eq!(allocated(), before);
    let (formed, resizer) = device.allocate_reserved(ALL, ALL, GRANULE, GRANULE).unwrap();
    assert!(formed.is_whole());
    assert_eq!(backed(), ALL, "a reservation formed with every granule backed");
    drop((formed, resizer));
}
