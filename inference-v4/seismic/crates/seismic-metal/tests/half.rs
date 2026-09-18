#![cfg(target_os = "macos")]
#[path = "../../../../validation/support/half_cases.rs"]
mod half_cases;
#[test]
#[ignore = "requires a Metal device"]
fn all_half_values_and_rounding_boundaries() {
    use seismic_metal::{msl, runtime::Device};
    let device = Device::open().unwrap();
    let info = device.info();
    half_cases::exercise(
        |lowered, values| {
            let emitted = msl::emit_with(
                lowered,
                seismic_metal::execution::Config {
                    max_threads_per_threadgroup: info.max_threads_per_threadgroup as i64,
                    max_threadgroup_bytes: info.max_threadgroup_bytes as i64,
                    ..Default::default()
                },
            )
            .unwrap();
            let pipeline = device.compile(emitted).unwrap();
            let buffers = values
                .iter()
                .map(|v| device.buffer_from(v).unwrap())
                .collect::<Vec<_>>();
            device
                .run(&pipeline, &buffers.iter().collect::<Vec<_>>(), &[], 1)
                .unwrap();
            for (value, buffer) in values.iter_mut().zip(&buffers) {
                *value = buffer.read(value.len());
            }
        },
        "metal",
    );
}
