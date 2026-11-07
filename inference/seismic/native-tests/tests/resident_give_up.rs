//! A resident launch whose threadgroups do not all arrive at a grid barrier:
//! the others' bounded waits give up, and the call fails as a lost device
//! within the bound, never hanging. Its own process, since a give-up stays
//! set on the barrier it was declared over.

#[cfg(target_os = "macos")]
#[test]
fn a_grid_barrier_that_not_every_threadgroup_reaches_loses_the_device() {
    use seismic::{BackendName, CallError, Devices, Element, ExecutionError, NativeSpecialization, Tensor};
    use seismic_native_tests::resident_reverse;
    use std::time::{Duration, Instant};

    let device = Devices::discover(&[]).unwrap().open_backend(BackendName::Metal).unwrap();
    let kernel = resident_reverse::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let skip = Tensor::from_host(&device, Element::i32(), &[1], &1i32.to_le_bytes()).unwrap();
    let n = 1u64 << 16;
    let zeros = vec![0u8; 4 * n as usize];
    let mut x = Tensor::from_host(&device, Element::f32(), &[n], &zeros).unwrap();
    let mut y = Tensor::from_host(&device, Element::f32(), &[n], &zeros).unwrap();
    let start = Instant::now();
    let result = kernel.call(resident_reverse::Args { skip: &skip, x: &mut x, y: &mut y }).map(|_| ());
    let elapsed = start.elapsed();
    println!("MEASURE give-up detected after {elapsed:?}");
    assert!(
        matches!(result, Err(CallError::Execution(ExecutionError::DeviceLost(_)))),
        "the call reports the give-up as a lost device: {result:?}"
    );
    assert!(elapsed < Duration::from_secs(2), "the give-up is detected within the bound: {elapsed:?}");
}
