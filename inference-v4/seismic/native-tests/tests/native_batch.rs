use seismic::{
    BackendName, DeviceCatalog, Element, NativeSpecialization, NativeTensorBatch, Tensor,
};
use seismic_native_tests::{scale_rows, scoped_scale};

#[test]
fn different_native_entries_complete_in_one_metal_batch() {
    let catalog = DeviceCatalog::discover().unwrap();
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let rows =
        scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let implementation = scoped_scale::native_implementation(&device)
        .unwrap()
        .unwrap();
    let defaults = implementation
        .default_specialization(&NativeSpecialization::new())
        .unwrap();
    let scoped = scoped_scale::native_for_device(&device, &defaults).unwrap();
    let first_values = [1.0f32, -2.0, 3.0, 4.0];
    let second_values = [2.0f32, 0.5, -3.0, 8.0, 1.0, -1.0, 0.0];
    let bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let first = Tensor::from_host(&device, Element::f32(), &[2, 2], &bytes(&first_values)).unwrap();
    let second = Tensor::from_host(&device, Element::f32(), &[7], &bytes(&second_values)).unwrap();
    // SAFETY: both authored entries write every f32 result element.
    let mut first_result =
        unsafe { Tensor::uninitialized(&device, Element::f32(), &[2, 2]) }.unwrap();
    let mut second_result =
        unsafe { Tensor::uninitialized(&device, Element::f32(), &[7]) }.unwrap();
    let mut batch = NativeTensorBatch::new(&device);
    batch
        .push(
            &rows,
            scale_rows::Args {
                x: &first,
                factor: 3.0,
            },
            scale_rows::OutputArgs {
                value: &mut first_result,
            },
        )
        .unwrap();
    batch
        .push(
            &scoped,
            scoped_scale::Args { x: &second },
            scoped_scale::OutputArgs {
                value: &mut second_result,
            },
        )
        .unwrap();
    batch.submit().unwrap().wait().unwrap();
    assert_eq!(
        first_result.read_to_host().unwrap(),
        bytes(&first_values.map(|value| value * 3.0))
    );
    assert_eq!(
        second_result.read_to_host().unwrap(),
        bytes(&second_values.map(|value| value * 2.0))
    );
}
