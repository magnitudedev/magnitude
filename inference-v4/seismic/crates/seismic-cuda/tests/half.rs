#[path = "../../../../validation/support/half_cases.rs"]
mod half_cases;
#[test]
#[ignore = "requires CUDA hardware"]
fn all_half_values_and_rounding_boundaries() {
    let device = seismic_cuda::Device::open(0).unwrap();
    half_cases::exercise(
        |lowered, buffers| {
            let mut kernel = device
                .compile(lowered, seismic_realization::Dispatch::ParallelRoot, 128)
                .unwrap();
            kernel
                .run(
                    &mut buffers
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    &[],
                )
                .unwrap();
        },
        "cuda",
    );
}
