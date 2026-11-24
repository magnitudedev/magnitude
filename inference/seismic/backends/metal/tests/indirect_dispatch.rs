//! Indirect launches on grids an earlier launch wrote, encoded as the direct
//! route's schedule places them after encoders of the other dispatch type:
//! a lone indirect launch (a serial encoder) after a group of several (a
//! concurrent one), and a group holding an indirect launch (a concurrent
//! encoder) after a lone launch (a serial one). On an M4 Pro (AGXMetalG16X,
//! macOS 15.7) each wrote or read address 0 inside the driver and killed the
//! process, as streamed experts' grid launches (`routed_copy`) did on a
//! served model's first request (`direct::Held`). Each indirect launch adds
//! one per threadgroup of its grid, so each count shows its grid was the one
//! written.
#![cfg(target_os = "macos")]

use seismic_compiler::executable::DeviceService;
use seismic_metal::toolchain::MetalToolchain;
use seismic_metal::{DeviceHandle, DirectBatch, DirectLaunch, MetalBuffer, MetalDevice};
use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};
use std::borrow::Borrow;
use std::sync::Arc;

const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
    kernel void grid(device uint* g [[buffer(0)]], uint i [[thread_position_in_grid]]) \
    { if (i == 0) { g[0] = 4; g[1] = 1; g[2] = 1; } }\n\
    kernel void count(device atomic_uint* x [[buffer(0)]], uint i [[thread_position_in_grid]], \
    uint l [[thread_position_in_threadgroup]]) { if (l == 0) atomic_fetch_add_explicit(x, 1 + i - i, memory_order_relaxed); }\n";

/// Threadgroups the `grid` kernel writes.
const GROUPS: u32 = 4;

#[test]
fn indirect_launches_after_encoders_of_the_other_type_run_their_written_grids() {
    let device = MetalDevice::open(
        DeviceHandle::system_default().unwrap(),
        Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam)),
    )
    .unwrap();
    let description = seismic_metal::profile::open_device(&device, &seismic_native_target::MemoryStore::default()).unwrap();
    let toolchain = MetalToolchain::new(device.clone(), description.device_info());
    let source = ProgramSource {
        text: SOURCE.into(),
        entries: vec![ProgramEntry::named("grid"), ProgramEntry::named("count")],
    };
    let pipelines = toolchain.compile(&source).unwrap().1;
    let entry: Arc<dyn seismic_memory::entry::Entry> =
        Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam));
    let allocate = |bytes| {
        let buffer = device.allocate(seismic_metal::topology::SHARED_KIND, bytes, 16, &entry).unwrap();
        device.write(&buffer, 0, &vec![0; bytes as usize]).unwrap();
        buffer
    };
    let (scratch, grids, counts) = ([allocate(16), allocate(16)], [allocate(16), allocate(16)], [allocate(16), allocate(16)]);
    let one = |batch: &mut DirectBatch, pipeline: usize, buffer: &MetalBuffer| {
        batch
            .encode(&DirectLaunch {
                pipeline: &pipelines[pipeline],
                buffers: &[(buffer, 0)],
                words: &[],
                scalar_results: None,
                threadgroups: [1, 1, 1],
                grid: None,
                threads_per_threadgroup: [64, 1, 1],
                threadgroup_bytes: 0,
            })
            .unwrap();
    };
    let indirect = |batch: &mut DirectBatch, which: usize| {
        let grid: &MetalBuffer = grids[which].borrow();
        batch
            .encode(&DirectLaunch {
                pipeline: &pipelines[1],
                buffers: &[(counts[which].borrow(), 0), (grid, 0)],
                words: &[],
                scalar_results: None,
                threadgroups: [GROUPS as u64, 1, 1],
                grid: Some((grid, 0)),
                threads_per_threadgroup: [64, 1, 1],
                threadgroup_bytes: 0,
            })
            .unwrap();
    };
    // The direct launches before each indirect one are of its own pipeline,
    // as the driver's failure needed.
    let mut batch = DirectBatch::new(&device).unwrap();
    // The grids, written on the device.
    batch.group(2);
    one(&mut batch, 0, grids[0].borrow());
    one(&mut batch, 0, grids[1].borrow());
    // A lone indirect launch after groups of several.
    batch.group(2);
    one(&mut batch, 1, scratch[0].borrow());
    one(&mut batch, 1, scratch[1].borrow());
    batch.group(1);
    indirect(&mut batch, 0);
    // A group holding an indirect launch after a lone launch.
    batch.group(1);
    one(&mut batch, 1, scratch[0].borrow());
    batch.group(2);
    indirect(&mut batch, 1);
    one(&mut batch, 1, scratch[1].borrow());
    batch.commit().finish().unwrap();
    for (which, count) in counts.iter().enumerate() {
        let mut bytes = [0u8; 4];
        device.read(count.borrow(), 0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), GROUPS, "indirect launch {which} ran its written grid");
    }
}

/// An indirect launch whose kernel only reads `[[thread_position_in_grid]]`
/// (such as `PROJECTION_ITEM_ARGUMENTS` relayout/coefficient passes) must run
/// its GPU-written 2D grid inside a concurrent encoder without silently
/// executing zero threads on AGX.
#[test]
fn indirect_thread_position_in_grid_only_kernel_runs_gpu_written_grid() {
    const TPG_SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
        kernel void write_grid(device uint* g [[buffer(0)]]) { g[0] = 3; g[1] = 8; g[2] = 1; }\n\
        kernel void mark(device uint* out [[buffer(0)]], uint3 p [[thread_position_in_grid]]) {\n\
            if (p.x == 0 && p.y == 0) out[0] = 99;\n\
            if (p.x == 512 && p.y == 7) out[1] = 77;\n\
        }\n";
    let device = MetalDevice::open(
        DeviceHandle::system_default().unwrap(),
        Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam)),
    )
    .unwrap();
    let description =
        seismic_metal::profile::open_device(&device, &seismic_native_target::MemoryStore::default()).unwrap();
    let toolchain = MetalToolchain::new(device.clone(), description.device_info());
    let source = ProgramSource {
        text: TPG_SOURCE.into(),
        entries: vec![ProgramEntry::named("write_grid"), ProgramEntry::named("mark")],
    };
    let pipelines = toolchain.compile(&source).unwrap().1;
    let entry: Arc<dyn seismic_memory::entry::Entry> =
        Arc::new(seismic_memory::entry::Tally::unbounded(seismic_topology::StoreIdentity::HostRam));
    let allocate = |bytes| {
        let buffer = device.allocate(seismic_metal::topology::SHARED_KIND, bytes, 16, &entry).unwrap();
        device.write(&buffer, 0, &vec![0; bytes as usize]).unwrap();
        buffer
    };
    let grid = allocate(16);
    let out = allocate(16);
    let mut batch = DirectBatch::new(&device).unwrap();
    batch.group(1);
    batch
        .encode(&DirectLaunch {
            pipeline: &pipelines[0],
            buffers: &[(grid.borrow(), 0)],
            words: &[],
            scalar_results: None,
            threadgroups: [1, 1, 1],
            grid: None,
            threads_per_threadgroup: [1, 1, 1],
            threadgroup_bytes: 0,
        })
        .unwrap();
    batch.group(1);
    batch
        .encode(&DirectLaunch {
            pipeline: &pipelines[1],
            buffers: &[(out.borrow(), 0)],
            words: &[],
            scalar_results: None,
            threadgroups: [3, 8, 1],
            grid: Some((grid.borrow(), 0)),
            threads_per_threadgroup: [256, 1, 1],
            threadgroup_bytes: 0,
        })
        .unwrap();
    batch.commit().finish().unwrap();
    let mut bytes = [0u8; 8];
    device.read(out.borrow(), 0, &mut bytes).unwrap();
    let first = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let last = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!((first, last), (99, 77), "thread_position_in_grid-only indirect launch must execute its GPU-written grid");
}
