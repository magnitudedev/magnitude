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

fn verify_native_graph(device: &seismic::Device) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = kernels::add_owned_f32::native_for_device(device)?;
    let mut graph = device.native_graph();
    let x_port = graph.port(seismic::Element::f32(), &[2, 2])?;
    let y_port = graph.port(seismic::Element::f32(), &[2, 2])?;
    let intermediate = graph.enqueue(
        &kernel,
        kernels::add_owned_f32::WorkflowArgs {
            x: x_port.tensor().into(),
            y: y_port.tensor().into(),
        },
    )?;
    let final_result = graph.enqueue(
        &kernel,
        kernels::add_owned_f32::WorkflowArgs {
            x: (&intermediate.value).into(),
            y: y_port.tensor().into(),
        },
    )?;
    graph.export(&final_result.value)?;
    let plan = graph.seal()?;
    assert_eq!(plan.workspace_bytes(), 16);
    assert_eq!(plan.output_bytes(), 16);
    let mut slot = plan.new_slot()?;
    let outputs = plan.new_outputs()?;
    let x = seismic::Tensor::from_host(
        device,
        seismic::Element::f32(),
        &[2, 2],
        &bytes(&[1.0, 2.0, 3.0, 4.0]),
    )?;
    let y = seismic::Tensor::from_host(
        device,
        seismic::Element::f32(),
        &[2, 2],
        &bytes(&[10.0, 20.0, 30.0, 40.0]),
    )?;
    let bound_plan = plan.bind_static(&[(&x_port, &x)])?;
    let mut bad = bound_plan.bindings();
    let wrong = seismic::Tensor::zeros(device, seismic::Element::f32(), &[4])?;
    bad.set(&y_port, &wrong)?;
    assert!(matches!(
        slot.attach(bad, outputs),
        Err(seismic::CallError::Workflow(
            seismic::WorkflowError::NativePortMismatch { .. }
        ))
    ));
    let mut bindings = bound_plan.bindings();
    bindings.set(&y_port, &y)?;
    let outputs = slot.attach(bindings, plan.new_outputs()?)?.run()?;
    drop(slot);
    let actual = outputs
        .exported(&final_result.value)
        .expect("exported result");
    assert_eq!(actual.read_to_host()?, bytes(&[21.0, 42.0, 63.0, 84.0]));
    let mut another_slot = plan.new_slot()?;
    let mut repeat = bound_plan.bindings();
    repeat.set(&y_port, &y)?;
    assert!(matches!(
        another_slot.attach(repeat, outputs),
        Err(seismic::CallError::Workflow(
            seismic::WorkflowError::NativeOutputLeaseConsumed
        ))
    ));

    let scale = kernels::scale_f32::native_for_device(device)?;
    // The two stages use one prepared kernel with different scalar ABI words.
    // A shared mutable words buffer would make both stages use the last factor.
    let mut repeated = device.native_graph();
    let repeated_input = repeated.input_for(&scale, "x", &[("M", 1), ("N", 2)])?;
    let mut doubled = repeated.local_for(&scale, "result", &[("M", 1), ("N", 2)])?;
    repeated.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: repeated_input.tensor().into(),
            factor: 2.0,
            result: doubled.tensor_mut().into(),
        },
    )?;
    let mut tripled = repeated.local_for(&scale, "result", &[("M", 1), ("N", 2)])?;
    repeated.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: doubled.tensor().into(),
            factor: 3.0,
            result: tripled.tensor_mut().into(),
        },
    )?;
    repeated.export(tripled.tensor())?;
    let repeated_plan = repeated.seal()?;
    let mut repeated_slot = repeated_plan.new_slot()?;
    repeated_slot.write_input(&repeated_input, &bytes(&[1.0, 2.0]))?;
    let repeated_outputs = repeated_slot
        .attach(repeated_plan.bindings(), repeated_plan.new_outputs()?)?
        .run()?;
    assert_eq!(
        repeated_outputs
            .exported(tripled.tensor())
            .expect("repeated-kernel graph export")
            .read_to_host()?,
        bytes(&[6.0, 12.0])
    );

    let mut owned_graph = device.native_graph();
    let input = owned_graph.input_for(&scale, "x", &[("M", 2), ("N", 2)])?;
    let mut output = owned_graph.local_for(&scale, "result", &[("M", 2), ("N", 2)])?;
    owned_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: input.tensor().into(),
            factor: 2.0,
            result: output.tensor_mut().into(),
        },
    )?;
    owned_graph.export(output.tensor())?;
    let owned_plan = owned_graph.seal()?;
    let mut small_graph = device.native_graph();
    let small_input = small_graph.input_for(&scale, "x", &[("M", 1), ("N", 2)])?;
    let mut small_output = small_graph.local_for(&scale, "result", &[("M", 1), ("N", 2)])?;
    small_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: small_input.tensor().into(),
            factor: 4.0,
            result: small_output.tensor_mut().into(),
        },
    )?;
    small_graph.export(small_output.tensor())?;
    let small_plan = small_graph.seal()?;
    let family = seismic::NativeGraphFamily::new(&[owned_plan.clone(), small_plan.clone()])?;
    assert_eq!(
        family.workspace_bytes(),
        owned_plan
            .workspace_bytes()
            .max(small_plan.workspace_bytes())
    );
    assert_eq!(family.output_bytes(), owned_plan.output_bytes());
    let mut family_slot = family.new_slot()?;
    let mut active = family_slot.activate(&owned_plan)?;
    active.write_input(&input, &bytes(&[2.0, 3.0, 4.0, 5.0]))?;
    let owned_outputs = active
        .attach(
            owned_plan.bindings(),
            family.new_output_slot()?.activate(&owned_plan)?,
        )?
        .run()?;
    drop(active);
    let retained = owned_outputs
        .exported(output.tensor())
        .expect("local export");
    assert_eq!(retained.read_to_host()?, bytes(&[4.0, 6.0, 8.0, 10.0]));
    drop(retained);
    let recycled = owned_outputs.recycle()?;
    let mut small_active = family_slot.activate(&small_plan)?;
    small_active.write_input(&small_input, &bytes(&[3.0, 7.0]))?;
    let small_outputs = small_active
        .attach(small_plan.bindings(), recycled.activate(&small_plan)?)?
        .run()?;
    let small_result = small_outputs
        .exported(small_output.tensor())
        .expect("small class export");
    assert_eq!(small_result.read_to_host()?, bytes(&[12.0, 28.0]));
    assert!(
        small_outputs.recycle().is_err(),
        "live export cannot recycle"
    );
    assert_eq!(small_result.read_to_host()?, bytes(&[12.0, 28.0]));

    let mut scratch_graph = device.native_graph();
    let scratch_input = scratch_graph.input_for(&scale, "x", &[("M", 1), ("N", 2)])?;
    let mut first_scratch = scratch_graph.local_for(&scale, "result", &[("M", 1), ("N", 2)])?;
    scratch_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: scratch_input.tensor().into(),
            factor: 10.0,
            result: first_scratch.tensor_mut().into(),
        },
    )?;
    let prewritten = scratch_graph.local_for(&scale, "x", &[("M", 1), ("N", 2)])?;
    scratch_graph.prewrite(&prewritten)?;
    let mut scratch_result = scratch_graph.local_for(&scale, "result", &[("M", 1), ("N", 2)])?;
    scratch_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: prewritten.tensor().into(),
            factor: 3.0,
            result: scratch_result.tensor_mut().into(),
        },
    )?;
    scratch_graph.export(scratch_result.tensor())?;
    let scratch_plan = scratch_graph.seal()?;
    assert_eq!(scratch_plan.workspace_bytes(), 24);
    let scratch_family = seismic::NativeGraphFamily::new(&[scratch_plan.clone()])?;
    let mut scratch_slot = scratch_family.new_slot()?;
    let mut scratch_active = scratch_slot.activate(&scratch_plan)?;
    let mut borrowed_local = scratch_active.local(&prewritten).expect("checked local");
    borrowed_local.write_from_host(&bytes(&[2.0, 5.0]))?;
    scratch_active.write_input(&scratch_input, &bytes(&[100.0, 100.0]))?;
    let scratch_outputs = scratch_active
        .attach(
            scratch_plan.bindings(),
            scratch_family.new_output_slot()?.activate(&scratch_plan)?,
        )?
        .run()?;
    let result = scratch_outputs
        .exported(scratch_result.tensor())
        .expect("scratch result");
    assert_eq!(result.read_to_host()?, bytes(&[6.0, 15.0]));
    assert_eq!(borrowed_local.read_to_host()?, bytes(&[2.0, 5.0]));
    drop(scratch_active);
    assert!(
        scratch_slot.activate(&scratch_plan).is_err(),
        "live local blocks scratch reuse"
    );
    drop(borrowed_local);
    assert!(scratch_slot.activate(&scratch_plan).is_ok());

    let mut reshape_graph = device.native_graph();
    let first = reshape_graph.input_for(&kernel, "x", &[("M", 2), ("N", 2)])?;
    let second = reshape_graph.input_for(&kernel, "y", &[("M", 2), ("N", 2)])?;
    let sum = reshape_graph.enqueue(
        &kernel,
        kernels::add_owned_f32::WorkflowArgs {
            x: first.tensor().into(),
            y: second.tensor().into(),
        },
    )?;
    let reshaped = sum.value.reshape(&[1, 4]);
    let mut scaled = reshape_graph.local_for(&scale, "result", &[("M", 1), ("N", 4)])?;
    reshape_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: (&reshaped).into(),
            factor: 3.0,
            result: scaled.tensor_mut().into(),
        },
    )?;
    let first_reshaped = first.tensor().reshape(&[1, 4]);
    let final_reshape = reshape_graph.enqueue(
        &kernel,
        kernels::add_owned_f32::WorkflowArgs {
            x: scaled.tensor().into(),
            y: (&first_reshaped).into(),
        },
    )?;
    reshape_graph.export(&final_reshape.value)?;
    let reshape_plan = reshape_graph.seal()?;
    // The two host inputs and intermediate need three 16-byte intervals;
    // the mutable local reuses a dead input interval in the same arena.
    assert_eq!(reshape_plan.workspace_bytes(), 48);
    let mut reshape_slot = reshape_plan.new_slot()?;
    reshape_slot.write_input(&first, &bytes(&[1.0, 2.0, 3.0, 4.0]))?;
    reshape_slot.write_input(&second, &bytes(&[10.0, 20.0, 30.0, 40.0]))?;
    let reshaped_outputs = reshape_slot
        .attach(reshape_plan.bindings(), reshape_plan.new_outputs()?)?
        .run()?;
    assert_eq!(
        reshaped_outputs
            .exported(&final_reshape.value)
            .expect("reshaped export")
            .read_to_host()?,
        bytes(&[34.0, 68.0, 102.0, 136.0])
    );

    // A later graph may write an entry's reserved export before that entry
    // runs. The output cannot be read through the public API until dispatch.
    let mut entry_graph = device.native_graph();
    let entry_input = entry_graph.input_for(&scale, "x", &[("M", 3), ("N", 2)])?;
    let mut entry_output = entry_graph.local_for(&scale, "result", &[("M", 3), ("N", 2)])?;
    entry_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: entry_input.tensor().into(),
            factor: 1.0,
            result: entry_output.tensor_mut().into(),
        },
    )?;
    entry_graph.export(entry_output.tensor())?;
    let entry_plan = entry_graph.seal()?;
    let mut overlay_graph = device.native_graph();
    let overlay_input = overlay_graph.port(seismic::Element::f32(), &[3, 2])?;
    let overlay_destination = overlay_graph.port(seismic::Element::f32(), &[3, 2])?;
    let input_first = overlay_input.tensor().slice_leading(0, 1);
    let mut destination_first = overlay_destination.tensor().slice_leading(0, 1);
    overlay_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: (&input_first).into(),
            factor: 10.0,
            result: (&mut destination_first).into(),
        },
    )?;
    let input_second = overlay_input.tensor().slice_leading(1, 2);
    let mut destination_second = overlay_destination.tensor().slice_leading(1, 2);
    overlay_graph.enqueue(
        &scale,
        kernels::scale_f32::WorkflowArgs {
            x: (&input_second).into(),
            factor: 100.0,
            result: (&mut destination_second).into(),
        },
    )?;
    let overlay_plan = overlay_graph.seal()?;
    assert_eq!(overlay_plan.workspace_bytes(), 0);
    assert_eq!(overlay_plan.output_bytes(), 0);
    let mut entry_slot = entry_plan.new_slot()?;
    entry_slot.write_input(&entry_input, &bytes(&[0.0, 0.0, 0.0, 0.0, 9.0, 10.0]))?;
    let reserved = entry_plan.new_outputs()?;
    assert!(reserved.exported(entry_output.tensor()).is_none());
    let mut overlay_bindings = overlay_plan.bindings();
    let overlay_source = seismic::Tensor::from_host(
        device,
        seismic::Element::f32(),
        &[3, 2],
        &bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]),
    )?;
    overlay_bindings.set(&overlay_input, &overlay_source)?;
    overlay_bindings.set_reserved_export(&overlay_destination, &reserved, entry_output.tensor())?;
    let mut overlay_slot = overlay_plan.new_slot()?;
    let ready_overlay = overlay_slot.attach(overlay_bindings, overlay_plan.new_outputs()?)?;
    let entry_outputs = entry_slot.attach(entry_plan.bindings(), reserved)?.run()?;
    ready_overlay.run()?;
    assert_eq!(
        entry_outputs
            .exported(entry_output.tensor())
            .expect("overlay destination export")
            .read_to_host()?,
        bytes(&[10.0, 20.0, 300.0, 400.0, 9.0, 10.0])
    );
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
    verify_native_graph(&device)?;
    println!(
        "native dense and polymorphic fixtures executed successfully on {}",
        device.info().name
    );
    Ok(())
}
