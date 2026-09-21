mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_f32(tensor: &seismic::Tensor) -> Result<f32, seismic::TensorError> {
    let bytes = tensor.read_to_host()?;
    Ok(f32::from_le_bytes(
        bytes.try_into().expect("one f32 result"),
    ))
}

fn verify_polymorphic_native(device: &seismic::Device) -> Result<(), Box<dyn std::error::Error>> {
    let f16 = seismic::Element::f16();
    let f16_input = seismic::Tensor::from_host(device, f16, &[1], &0x4200u16.to_le_bytes())?;
    let mut f16_result = seismic::Tensor::zeros(device, seismic::Element::f32(), &[1])?;
    kernels::read_first::native_for_device_with(device, kernels::read_first::Elements { E: f16 })?
        .call(kernels::read_first::Args {
            x: &f16_input,
            result: &mut f16_result,
        })?;
    assert_eq!(read_f32(&f16_result)?, 3.0);

    let q8g32 = seismic::Element::named("q8g32").expect("q8g32 representation");
    let mut q8g32_bytes = vec![0u8; 36];
    q8g32_bytes[0] = 3;
    q8g32_bytes[32..36].copy_from_slice(&2.0f32.to_le_bytes());
    let q8g32_input = seismic::Tensor::from_host(device, q8g32, &[32], &q8g32_bytes)?;
    let mut q8g32_result = seismic::Tensor::zeros(device, seismic::Element::f32(), &[1])?;
    kernels::read_first::native_for_device_with(
        device,
        kernels::read_first::Elements { E: q8g32 },
    )?
    .call(kernels::read_first::Args {
        x: &q8g32_input,
        result: &mut q8g32_result,
    })?;
    assert_eq!(read_f32(&q8g32_result)?, 6.0);
    Ok(())
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

    let owned = kernels::add_owned_f32::native_for_device(&device)?
        .call(kernels::add_owned_f32::Args { x: &x, y: &y })?;
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
    verify_polymorphic_native(&device)?;
    println!(
        "native dense and polymorphic fixtures executed successfully on {}",
        device.info().name
    );
    Ok(())
}
