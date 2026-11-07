//! A resident launch (`launch K resident (BARRIER)`): its threadgroups are
//! on the device at once, so its phases meet at grid barriers
//! (`seismic_grid::sync`), each reading what other threadgroups wrote in the
//! phase before, and its barrier is restored for the next launch over it.

#[cfg(target_os = "macos")]
#[test]
fn a_resident_launchs_phases_meet_at_grid_barriers_launch_after_launch() {
    use seismic::{BackendName, Devices, Element, NativeSpecialization, Tensor};
    use seismic_native_tests::resident_reverse;

    let device = Devices::discover(&[]).unwrap().open_backend(BackendName::Metal).unwrap();
    let kernel = resident_reverse::native_for_device(&device, &NativeSpecialization::new()).unwrap();
    let skip = Tensor::from_host(&device, Element::i32(), &[1], &0i32.to_le_bytes()).unwrap();
    let read = |tensor: &Tensor| {
        tensor
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    // Grids of one threadgroup, of fewer than RESIDENT, and of more work than
    // RESIDENT holds, so threadgroups own several elements; each launch over
    // the barrier the one before restored.
    for n in [1u64, 64, 1000, 1 << 20, 1 << 20] {
        let zeros = vec![0u8; 4 * n as usize];
        let mut x = Tensor::from_host(&device, Element::f32(), &[n], &zeros).unwrap();
        let mut y = Tensor::from_host(&device, Element::f32(), &[n], &zeros).unwrap();
        kernel
            .call(resident_reverse::Args { skip: &skip, x: &mut x, y: &mut y })
            .unwrap_or_else(|error| panic!("N {n}: {error}"));
        let (x, y) = (read(&x), read(&y));
        for i in 0..n as usize {
            assert_eq!(y[i], 2.0 * (n as usize - i) as f32, "N {n}: y[{i}]");
            assert_eq!(x[i], 2.0 * (i + 1) as f32 + 1.0, "N {n}: x[{i}]");
        }
    }
}
