//! A timed batch whose every timestamp page is in flight behind a host
//! gate of its own not yet open fails to encode rather than wait for a page
//! that cannot free. The pool is the process's (`LaunchTimestamps`), so
//! this test has a process to itself (an integration test binary).
#![cfg(target_os = "macos")]

use seismic_compiler::executable::DeviceService;
use seismic_metal::toolchain::MetalToolchain;
use seismic_metal::{DeviceHandle, DirectBatch, DirectLaunch, LaunchTimestamps, MetalDevice, SharedEvent};
use seismic_native_target::{ProgramEntry, ProgramSource, Toolchain};

#[test]
fn a_timed_batch_behind_its_own_shut_gate_fails_when_the_pool_is_exhausted() {
    let device = MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap();
    let description = seismic_metal::profile::open_device(&device).unwrap();
    let toolchain = MetalToolchain::new(device.clone(), description.facts());
    let pipelines = toolchain
        .compile(
            &ProgramSource {
                text: "#include <metal_stdlib>\nusing namespace metal;\n\
                       kernel void bump(device float* x [[buffer(0)]], \
                       uint i [[thread_position_in_grid]]) { x[i] += 1; }\n"
                    .into(),
                entries: vec![ProgramEntry::named("bump")],
            },
            None,
        )
        .unwrap();
    let values = device.allocate(256, 16).unwrap();
    let launch = DirectLaunch {
        pipeline: &pipelines[0],
        buffers: &[(&values, 0)],
        words: &[],
        scalar_results: None,
        threadgroups: [1, 1, 1],
        grid: None,
        threads_per_threadgroup: [64, 1, 1],
        threadgroup_bytes: 0,
    };
    let timestamps = LaunchTimestamps::shared(&device).unwrap();
    let gate = SharedEvent::new(&device).unwrap();
    let mut batch = DirectBatch::timed(&device, timestamps).unwrap();
    batch.wait_event(&gate, 1).unwrap();
    // Every page holds 2,048 launches' samples, and Metal allows a process
    // 32 pages (`tests/sample_buffers.rs`): far fewer launches than this
    // exhaust the pool.
    let error = (0..1 << 20)
        .find_map(|_| batch.encode(&launch).err())
        .expect("encoding behind a shut gate exhausts the pool");
    assert!(
        error.to_string().contains("timestamp pool exhausted"),
        "{error}"
    );
    // Opened, the gate lets the committed work complete and free its pages.
    gate.signal(1);
    if let Some(submission) = batch.abandon() {
        submission.finish().unwrap();
    }
}
