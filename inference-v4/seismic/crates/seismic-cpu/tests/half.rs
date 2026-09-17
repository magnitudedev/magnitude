#[path = "../../../../validation/support/half_cases.rs"]
mod half_cases;
#[test]
fn all_half_values_and_rounding_boundaries() {
    half_cases::exercise(
        |lowered, buffers| {
            let mut kernel = seismic_cpu::compile(lowered).unwrap();
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
        "cpu",
    );
}
