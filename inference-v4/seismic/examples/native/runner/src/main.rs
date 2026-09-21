mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = seismic::DeviceCatalog::discover()?;
    let device = catalog.open_backend(seismic::BackendName::Metal)?;
    let x = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[2, 2],
        &bytes(&[1.0, 2.0, 3.0, 4.0]),
    )?;
    let y = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[2, 2],
        &bytes(&[10.0, 20.0, 30.0, 40.0]),
    )?;
    let mut result = seismic::Tensor::zeros(&device, seismic::Element::f32(), &[2, 2])?;

    let kernel = kernels::add_f32::native_for_device(&device)?;
    let mut wrong_shape = seismic::Tensor::zeros(&device, seismic::Element::f32(), &[4])?;
    assert!(matches!(
        kernel.call(kernels::add_f32::Args {
            x: &x,
            y: &y,
            result: &mut wrong_shape,
        }),
        Err(seismic::CallError::Invocation(_))
    ));
    kernel.call(kernels::add_f32::Args {
        x: &x,
        y: &y,
        result: &mut result,
    })?;

    let actual = result.read_to_host()?;
    assert_eq!(actual, bytes(&[11.0, 22.0, 33.0, 44.0]));

    let owned = kernels::add_owned_f32::native_for_device(&device)?.call(
        kernels::add_owned_f32::Args { x: &x, y: &y },
    )?;
    assert_eq!(
        owned.value.read_to_host()?,
        bytes(&[11.0, 22.0, 33.0, 44.0])
    );

    let mut scaled = seismic::Tensor::zeros(&device, seismic::Element::f32(), &[2, 2])?;
    kernels::scale_f32::native_for_device(&device)?.call(kernels::scale_f32::Args {
        x: &x,
        factor: 2.5,
        result: &mut scaled,
    })?;
    assert_eq!(scaled.read_to_host()?, bytes(&[2.5, 5.0, 7.5, 10.0]));
    println!("native add_f32 executed successfully on {}", device.info().name);
    Ok(())
}
