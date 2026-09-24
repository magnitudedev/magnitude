//! The direct-native route on every backend this host can open: static
//! dimensions, tuning parameters, `where`, scratch, several launches, shared
//! memory, graphs with host-written inputs, asynchronous submission,
//! measurement and exhaustive tuning.

use seismic::{
    Availability, BackendName, CallError, Device, DeviceCatalog, Element, InvocationError, MeasureOptions,
    NativeGraphFamily, NativeSpecialization, Outcome, Tensor, TuningPoint, Validation,
};
use seismic_native_tests::{scale_rows, split_sum};

/// Every backend the catalog reports available on this host; each must open.
fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let topology = catalog.topology();
    let devices = [BackendName::Cpu, BackendName::Metal, BackendName::Cuda]
        .into_iter()
        .filter(|backend| {
            topology.devices().iter().any(|device| {
                device.backend == *backend && matches!(device.availability, Availability::Available)
            })
        })
        .map(|backend| {
            catalog
                .open_backend(backend)
                .unwrap_or_else(|error| panic!("available {backend:?} device must open: {error}"))
        })
        .collect::<Vec<_>>();
    eprintln!(
        "native route backends: {:?}",
        devices.iter().map(Device::backend).collect::<Vec<_>>()
    );
    devices
}

fn f32_tensor(device: &Device, extents: &[u64], values: &[f32]) -> Tensor {
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), extents, &bytes).expect("host tensor")
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .expect("host read")
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("f32 word")))
        .collect()
}

/// Values whose every partial sum is exact in f32, so all configurations
/// agree bit for bit with the ordered reference.
fn exact_values(n: usize) -> Vec<f32> {
    (0..n).map(|index| ((index % 7) as f32 - 3.0) * 0.25).collect()
}

fn statics(n: u64) -> NativeSpecialization {
    NativeSpecialization::new().with_static("N", n)
}

#[test]
fn every_admissible_split_sum_configuration_matches_the_reference() {
    for device in devices() {
        let n = 1000usize;
        let values = exact_values(n);
        let expected: f32 = values.iter().sum();
        let x = f32_tensor(&device, &[n as u64], &values);
        let implementation = split_sum::native_implementation(&device)
            .expect("bundle")
            .expect("split_sum has an implementation on every backend");
        let configurations = implementation.admissible(&statics(n as u64)).expect("statics");
        // Three part counts by two widths, all admissible at N = 1000.
        assert_eq!(configurations.len(), 6, "{}", device.backend().as_str());
        for configuration in configurations {
            let kernel = split_sum::native_for_device(&device, &configuration)
                .unwrap_or_else(|error| panic!("{configuration:?} on {:?}: {error}", device.backend()));
            let result = kernel.call(split_sum::Args { x: &x }).expect("call");
            assert_eq!(
                read_f32(&result.value),
                [expected],
                "{configuration:?} on {:?}",
                device.backend()
            );
        }
    }
}

#[test]
fn where_filters_configurations_at_small_static_values() {
    for device in devices() {
        let implementation = split_sum::native_implementation(&device).unwrap().unwrap();
        let admissible = implementation.admissible(&statics(2)).unwrap();
        // PARTS <= N keeps parts 1 and 2 at N = 2.
        assert_eq!(admissible.len(), 4);
        assert!(admissible.iter().all(|configuration| configuration.param("PARTS") != Some(4)));
        let inadmissible = statics(2).with_param("PARTS", 4).with_param("WIDTH", 32);
        assert!(split_sum::native_for_device(&device, &inadmissible).is_err());
    }
}

#[test]
fn specialization_errors_are_typed_at_preparation() {
    for device in devices() {
        let missing_static = NativeSpecialization::new()
            .with_param("PARTS", 1)
            .with_param("WIDTH", 32);
        assert!(split_sum::native_for_device(&device, &missing_static).is_err());
        let outside = statics(64).with_param("PARTS", 3).with_param("WIDTH", 32);
        let error = split_sum::native_for_device(&device, &outside)
            .err()
            .expect("3 is outside PARTS's domain");
        assert!(error.to_string().contains("does not admit 3"), "{error}");
        let missing_param = statics(64).with_param("PARTS", 1);
        assert!(split_sum::native_for_device(&device, &missing_param).is_err());
    }
}

#[test]
fn a_call_whose_static_dimension_differs_is_rejected() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 2).with_param("WIDTH", 32),
        )
        .unwrap();
        let x = f32_tensor(&device, &[63], &exact_values(63));
        match kernel.call(split_sum::Args { x: &x }) {
            Err(CallError::Invocation(InvocationError::StaticDimension {
                dimension,
                expected,
                ..
            })) => {
                assert_eq!(dimension, "N");
                assert_eq!(expected, 64);
            }
            other => panic!("expected a static-dimension error, got {:?}", other.err()),
        }
    }
}

/// A graph whose input the host writes: two runs submitted back to back
/// without waiting must each read their own input.
#[test]
fn graph_runs_submit_without_waiting_and_keep_their_inputs() {
    for device in devices() {
        let (m, n) = (5u64, 37u64);
        let kernel =
            scale_rows::native_for_device(&device, &NativeSpecialization::new().with_param("ROWS", 2))
                .unwrap();
        let mut graph = device.native_graph();
        let input = graph.input_for(&kernel, "x", &[("M", m), ("N", n)]).unwrap();
        let first = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: input.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap();
        let second = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: (&first.value).into(),
                    factor: 3.0,
                },
            )
            .unwrap();
        graph.export(&second.value).unwrap();
        let plan = graph.seal().unwrap();
        assert!(plan.upload_bytes() >= m * n * 4);
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot().unwrap();
        let mut runs = Vec::new();
        for run in 0..3u64 {
            let values = (0..m * n)
                .map(|index| (index + run * 1000) as f32)
                .collect::<Vec<_>>();
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            let outputs = family.new_output_slot().unwrap().activate(&plan).unwrap();
            let mut active = slot.activate(&plan).unwrap();
            active.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = active
                .attach(plan.bindings(), outputs)
                .unwrap()
                .submit()
                .unwrap();
            runs.push((values, outputs, completion));
        }
        for (values, outputs, completion) in runs {
            completion.wait().unwrap();
            let result = read_f32(&outputs.exported(&second.value).unwrap());
            let expected = values.iter().map(|value| value * 6.0).collect::<Vec<_>>();
            assert_eq!(result, expected, "{:?}", device.backend());
        }
    }
}

#[test]
fn measurement_reports_device_time() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(4096).with_param("PARTS", 4).with_param("WIDTH", 64),
        )
        .unwrap();
        let rotation = (0..3)
            .map(|_| f32_tensor(&device, &[4096], &exact_values(4096)))
            .collect::<Vec<_>>();
        let measurement = kernel
            .measure(
                rotation.iter().map(|x| split_sum::Args { x }).collect(),
                &MeasureOptions {
                    samples: 3,
                    min_sample_seconds: 0.0005,
                },
            )
            .unwrap();
        assert_eq!(measurement.samples.len(), 3);
        assert!(measurement.median > 0.0, "{:?}", device.backend());
        assert!(measurement.rotation_bytes >= 3 * 4096 * 4);
    }
}

#[test]
fn tuning_chooses_measured_configurations_with_one_arithmetic_assignment() {
    for device in devices() {
        let n = 4096u64;
        let inputs = (0..2)
            .map(|_| f32_tensor(&device, &[n], &exact_values(n as usize)))
            .collect::<Vec<_>>();
        let points = vec![
            TuningPoint {
                label: "short".into(),
                weight: 1.0,
                rotation: inputs.iter().map(|x| split_sum::Args { x }).collect(),
            },
            TuningPoint {
                label: "long".into(),
                weight: 3.0,
                rotation: inputs.iter().map(|x| split_sum::Args { x }).collect(),
            },
        ];
        let result = split_sum::native_tune(
            &device,
            &statics(n),
            points,
            Validation::BitExact,
            MeasureOptions {
                samples: 3,
                min_sample_seconds: 0.0005,
            },
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        assert_eq!(result.configurations.len(), 6);
        assert!(result
            .configurations
            .iter()
            .all(|record| matches!(record.outcome, Outcome::Measured { .. })));
        assert_eq!(result.chosen.len(), 2);
        assert_eq!(result.chosen[0].params["PARTS"], result.chosen[1].params["PARTS"]);
        assert!(result
            .configurations
            .iter()
            .any(|record| record.configuration == result.overall));
        // The tuned configuration prepares and runs.
        let kernel = split_sum::native_for_device(&device, &result.overall.specialization()).unwrap();
        let value = kernel.call(split_sum::Args { x: &inputs[0] }).unwrap().value;
        assert_eq!(read_f32(&value), [exact_values(n as usize).iter().sum::<f32>()]);
    }
}
