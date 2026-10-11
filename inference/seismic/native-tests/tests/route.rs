//! The direct-native route on every backend this host can open: static
//! dimensions, tuning parameters, `where`, scratch, several launches, shared
//! memory, conditional (`when`) launches and scratch, graphs with
//! host-written inputs, asynchronous submission, measurement and tuning.

use seismic::{
    Availability, BackendName, CallError, Device, DeviceCatalog, Element, Exclusion,
    InvocationError, MeasureOptions, NativeGraphFamily, NativeSpecialization,
    NumericalRequirement, Outcome, SearchPlan, SearchSettings, Stage, Strategy,
    SurveyPlan, Tensor, TraceDetail, TuneError, TuningInitializer, TuningMethod, TuningPoint,
    UnitPlan,
};
use seismic_native_tests::{accumulate, gated_sum, scale_rows, scoped_scale, split_sum};
use std::time::Duration;

/// The range every non-empty scratch arena reserves at its start for `sync`
/// scratch (the runtime's `SYNC_SCRATCH_BYTES`).
const SYNC_RESERVE: u64 = 64 * 1024;

/// A search whose time outlasts its convergence on the test entries.
fn search() -> Strategy {
    Strategy::Search(SearchPlan {
        allowance: Duration::from_secs(3600),

        settings: SearchSettings {
            error_rate: 0.05,
        },
        // Long enough that a sample of these µs-scale kernels measures the
        // kernel rather than the fixed cost of its submission.
        min_sample_seconds: 0.005,
        start: Vec::new(),
    })
}

/// [`search`], hinted each of `hints` (a parameter at a value, the others at
/// their defaults). Hints are measured in the unit's first session, and a
/// unit witnesses the challengers of its first session before it times
/// them: it has observed neither test's cost to order them by. So a hinted
/// candidate is witnessed, whichever order the unit's observations give the
/// challengers it reaches afterwards.
fn search_hinted(hints: &[(&str, u64)]) -> Strategy {
    let Strategy::Search(mut plan) = search() else {
        unreachable!()
    };
    plan.start = hints
        .iter()
        .map(|(name, value)| [((*name).to_owned(), *value)].into_iter().collect())
        .collect();
    Strategy::Search(plan)
}

/// Every backend the catalog reports available on this host; each must open.
fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let topology = catalog.topology();
    let devices = [
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ]
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
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
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
    (0..n)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.25)
        .collect()
}

/// The declared parameters of a configuration: those Seismic does not own
/// (Seismic-owned names contain `.`).
fn declared(
    params: &std::collections::BTreeMap<String, u64>,
) -> std::collections::BTreeMap<String, u64> {
    params
        .iter()
        .filter(|(name, _)| !name.contains('.'))
        .map(|(name, value)| (name.clone(), *value))
        .collect()
}

fn statics(n: u64) -> NativeSpecialization {
    NativeSpecialization::new().with_static("N", n)
}

#[test]
fn metal_launch_scopes_form_and_execute_independently() {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let implementation = scoped_scale::native_implementation(&device)
        .unwrap()
        .expect("Metal has the scoped fixture");
    let defaults = implementation
        .default_specialization(&NativeSpecialization::new())
        .unwrap();
    for (rows, launch, code, width) in [
        (7usize, 0, 1, 32),
        (7, 0, 2, 64),
        (35, 1, 4, 64),
        (35, 1, 8, 128),
    ] {
        let specialization = defaults
            .clone()
            .with_launch_param(launch, "ROWS", code)
            .with_launch_param(launch, "WIDTH", width);
        let kernel = scoped_scale::native_for_device(&device, &specialization)
            .expect("each scoped Metal specialization forms");
        let values = exact_values(rows);
        let x = f32_tensor(&device, &[rows as u64], &values);
        let result = kernel
            .call(scoped_scale::Args { x: &x })
            .expect("scoped call");
        let expected = values.iter().map(|value| value * 2.0).collect::<Vec<_>>();
        assert_eq!(read_f32(&result.value), expected);
    }
}

/// Serving prepares the configuration tuning chose as tuning formed it: no
/// program is formed a second time.
#[test]
fn serving_the_tuned_choice_forms_nothing_again() {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let small = f32_tensor(&device, &[7], &exact_values(7));
    let large = f32_tensor(&device, &[35], &exact_values(35));
    let point = |label: &str, x| TuningPoint {
        label: label.into(),
        weight: 1.0,
        class: None,
        cost: 1.0,
        workload: seismic::WorkloadRotation::All,
        rotation: vec![scoped_scale::Args { x }],
        initialize: None,
        written: Default::default(),
    };
    let result = scoped_scale::native_tune(
        &device,
        &NativeSpecialization::new(),
        vec![point("m7", &small), point("m35", &large)],
        NumericalRequirement::BitExact,
        search(),
    )
    .expect("factored Metal tuning");
    let formed = device.programs_formed();
    scoped_scale::native_for_device(&device, &result.overall.specialization())
        .expect("the chosen configuration prepares");
    assert_eq!(device.programs_formed(), formed);
}

#[test]
fn metal_scoped_tuning_searches_launches_separately() {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let small = f32_tensor(&device, &[7], &exact_values(7));
    let large = f32_tensor(&device, &[35], &exact_values(35));
    let points = vec![
        TuningPoint {
            label: "m7".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![scoped_scale::Args { x: &small }],
            initialize: None,
            written: Default::default(),
        },
        TuningPoint {
            label: "m35".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![scoped_scale::Args { x: &large }],
            initialize: None,
            written: Default::default(),
        },
    ];
    let result = scoped_scale::native_tune(
        &device,
        &NativeSpecialization::new(),
        points,
        NumericalRequirement::BitExact,
        search(),
    )
    .expect("factored Metal tuning");
    // The search ends complete having asked for part of the two groups'
    // sixteen combinations: the nearby changes of each launch's
    // parameters and what leads from them.
    let TuningMethod::Search {
        groups: 2,
        candidates,
        complete: true,
    } = result.method
    else {
        panic!("{:?}", result.method);
    };
    assert_eq!(candidates + 1, result.configurations.len());
    assert!((4..=15).contains(&candidates), "{candidates}");
    assert_eq!(result.overall.launches.len(), 2);
    let kernel =
        scoped_scale::native_for_device(&device, &result.overall.specialization()).unwrap();
    for (tensor, count) in [(&small, 7), (&large, 35)] {
        let result = kernel.call(scoped_scale::Args { x: tensor }).unwrap();
        assert_eq!(
            read_f32(&result.value),
            exact_values(count)
                .iter()
                .map(|value| value * 2.0)
                .collect::<Vec<_>>()
        );
    }
}

/// A unit's compiles need no device and nothing of its search: the
/// canonical program and each challenger piece's launch sources compile on
/// another thread, and the unit's work then compiles nothing itself.
#[test]
fn metal_compiles_run_ahead_on_another_thread_and_the_piece_then_compiles_nothing() {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let small = f32_tensor(&device, &[7], &exact_values(7));
    let large = f32_tensor(&device, &[35], &exact_values(35));
    let point = |label: &str, x| TuningPoint {
        label: label.into(),
        weight: 1.0,
        class: None,
        cost: 1.0,
        workload: seismic::WorkloadRotation::All,
        rotation: vec![scoped_scale::Args { x }],
        initialize: None,
        written: Default::default(),
    };
    let mut source = vec![point("m7", &small), point("m35", &large)];
    let mut unit = scoped_scale::native_search(
        &device,
        &NativeSpecialization::new(),
        NumericalRequirement::BitExact,
    )
    .unwrap();
    let grounding = unit
        .grounding(&mut source)
        .unwrap()
        .expect("the canonical program is not formed yet");
    let grounded = std::thread::spawn(move || grounding.run()).join().unwrap();
    unit.grounded(grounded).unwrap();
    assert!(unit.grounding(&mut source).unwrap().is_none());
    let Strategy::Search(plan) = search() else {
        unreachable!()
    };
    unit.plan(UnitPlan {
        settings: plan.settings,
        min_sample_seconds: 0.0002,
        start: Vec::new(),
    });
    let mut ahead = 0;
    // After each measurement: the configurations measured so far and
    // Seismic's charges on the device.
    let mut held = Vec::new();
    // What is charged with the points discovered and nothing measured: the
    // unit's inputs, its result storage and the one kernel it serves.
    let mut discovered = None;
    while let Some(work) = unit.next() {
        if work.stage == Stage::Measure && discovered.is_none() {
            discovered = Some(device.memory_usage().charged);
        }
        let ended = if let Some(forming) = unit.forming() {
            assert_eq!(work.stage, Stage::Measure);
            assert_ne!(work.forming, seismic::Quote::default());
            let formed = std::thread::spawn(move || forming.run()).join().unwrap();
            unit.formed(formed).unwrap();
            // The piece is the same, less its compile.
            let rest = unit.next().unwrap();
            assert_eq!(rest.stage, Stage::Measure);
            assert_eq!(rest.forming, seismic::Quote::default());
            assert!(unit.forming().is_none());
            let performed = unit.perform(&mut source).unwrap();
            assert_eq!(performed.spent.compiling, 0.);
            ahead += 1;
            // The piece's high-water mark is no less than either end.
            assert!(performed.memory.peak_bytes >= performed.memory.ended_bytes);
            assert!(performed.memory.peak_bytes >= performed.memory.began_bytes);
            performed.memory.ended_bytes
        } else {
            assert_eq!(work.forming, seismic::Quote::default());
            unit.perform(&mut source).unwrap().memory.ended_bytes
        };
        // What a piece holds is the device's charges and the process's
        // host bytes: no less than the charges.
        let charged = device.memory_usage().charged;
        assert!(ended >= charged);
        if work.stage == Stage::Measure {
            held.push((unit.standing().measured, charged));
        }
    }
    assert!(ahead > 0, "a challenger's launch sources compiled ahead");
    // A measured configuration that neither serves nor leads owns nothing
    // on the device. A formed kernel of this fixture charges the device
    // one byte, its scalars; a unit holds between pieces the kernels of
    // its served choice and of a resolved and a nominal leader: one, two
    // or three of them. The challengers write into result storage of
    // their own, one f32 per row of each point, allocated once. So
    // whatever was measured, a measurement ends with the bytes of the
    // discovered unit, that storage and at most two more. The search
    // measures the defaults, the four nearby changes of its two launches
    // and what leads from them: kept kernels would end it at four bytes
    // more at the least.
    let discovered = discovered.expect("the unit measured challengers") + 4 * (7 + 35);
    let last = *held.last().unwrap();
    assert!(last.0 >= 5, "{held:?}");
    assert!(
        held.iter()
            .all(|(_, charged)| (discovered..=discovered + 2).contains(charged)),
        "device bytes follow the configurations measured: {held:?} from {discovered}"
    );
    // With no work left the unit gives up its points' inputs and result
    // storage, and its record stands without them.
    assert!(unit.holds_inputs());
    unit.release(&mut source);
    assert!(!unit.holds_inputs());
    assert!(device.memory_usage().charged < discovered, "{held:?}");
    let result = unit.terminal().unwrap();
    let TuningMethod::Search {
        groups: 2,
        candidates,
        complete: true,
    } = result.method
    else {
        panic!("{:?}", result.method);
    };
    // Every configuration the search asked for was measured or excluded,
    // and each group has four assignments of which the defaults are one:
    // at most the two groups' other three each and their combinations.
    assert_eq!(candidates + 1, result.configurations.len());
    assert!((4..=15).contains(&candidates), "{candidates}");
}

#[test]
fn metal_scoped_tuning_without_budget_serves_its_formed_defaults() {
    let catalog = DeviceCatalog::discover().expect("device discovery");
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let small = f32_tensor(&device, &[7], &exact_values(7));
    let large = f32_tensor(&device, &[35], &exact_values(35));
    let points = vec![
        TuningPoint {
            label: "m7".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![scoped_scale::Args { x: &small }],
            initialize: None,
            written: Default::default(),
        },
        TuningPoint {
            label: "m35".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![scoped_scale::Args { x: &large }],
            initialize: None,
            written: Default::default(),
        },
    ];
    let Strategy::Search(mut plan) = search() else {
        unreachable!()
    };
    plan.allowance = Duration::ZERO;
    let result = scoped_scale::native_tune(
        &device,
        &NativeSpecialization::new(),
        points,
        NumericalRequirement::BitExact,
        Strategy::Search(plan),
    )
    .unwrap();
    assert!(matches!(
        result.disposition,
        seismic::TuningDisposition::CanonicalFallback { .. }
    ));
    assert!(result.configurations.is_empty());
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
        let configurations = implementation
            .admissible(&statics(n as u64))
            .expect("statics");
        // Three part counts by two widths, all admissible at N = 1000.
        assert_eq!(configurations.len(), 6, "{}", device.backend().as_str());
        for configuration in configurations {
            let kernel =
                split_sum::native_for_device(&device, &configuration).unwrap_or_else(|error| {
                    panic!("{configuration:?} on {:?}: {error}", device.backend())
                });
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
        assert!(
            admissible
                .iter()
                .all(|configuration| configuration.param("PARTS") != Some(4))
        );
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
        let kernel = scale_rows::native_for_device(
            &device,
            &NativeSpecialization::new().with_param("ROWS", 2),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let input = graph
            .input_for(&kernel, "x", &[("M", m), ("N", n)])
            .unwrap();
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
        // One upload region per run kept in flight below.
        let mut slot = family.new_slot(3).unwrap();
        let mut runs = Vec::new();
        for run in 0..3u64 {
            let values = (0..m * n)
                .map(|index| (index + run * 1000) as f32)
                .collect::<Vec<_>>();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
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
fn tuning_searches_from_the_defaults_and_validates_its_choice() {
    for device in devices() {
        let n = 4096u64;
        let inputs = (0..2)
            .map(|_| f32_tensor(&device, &[n], &exact_values(n as usize)))
            .collect::<Vec<_>>();
        fn points(inputs: &[Tensor]) -> Vec<TuningPoint<'_, split_sum::Entry>> {
            vec![
                TuningPoint {
                    label: "short".into(),
                    weight: 1.0,
                    class: None,
                    cost: 1.0,
                    workload: seismic::WorkloadRotation::All,
                    rotation: inputs.iter().map(|x| split_sum::Args { x }).collect(),
                    initialize: None,
                    written: Default::default(),
                },
                TuningPoint {
                    label: "long".into(),
                    weight: 3.0,
                    class: None,
                    cost: 1.0,
                    workload: seismic::WorkloadRotation::All,
                    rotation: inputs.iter().map(|x| split_sum::Args { x }).collect(),
                    initialize: None,
                    written: Default::default(),
                },
            ]
        }
        let result = split_sum::native_tune(
            &device,
            &statics(n),
            points(&inputs),
            NumericalRequirement::BitExact,
            search(),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        // With time to spare the search ends on its own, having reached
        // each configuration at most once, starting from the defaults. The
        // domain is the declared one (three part counts by two widths) and,
        // on a CPU device, its participant counts per launch and tiers.
        let domain = result
            .parameters
            .iter()
            .map(|parameter| parameter.values.len())
            .product::<usize>();
        assert_eq!(domain % 6, 0);
        assert!(result.configurations.len() <= domain);
        // A program formed whole is one group of the search, and what it
        // records means what it does for launches formed separately: every
        // configuration it asked its domain for was measured.
        let TuningMethod::Search {
            groups: 1,
            candidates,
            complete: true,
        } = result.method
        else {
            panic!("{:?}", result.method);
        };
        assert_eq!(candidates + 1, result.configurations.len());
        assert_eq!(
            declared(&result.configurations[0].configuration.params),
            [("PARTS".to_owned(), 1), ("WIDTH".to_owned(), 32)]
                .into_iter()
                .collect()
        );
        assert!(
            result.configurations[0]
                .configuration
                .params
                .iter()
                .all(|(name, value)| !name.starts_with("cpu.") || *value == 0)
        );
        // The chosen configuration is measured and admitted.
        assert!(result.configurations.iter().any(|record| {
            record.configuration == result.overall
                && matches!(record.outcome, Outcome::Measured { .. })
                && result.admitted(record).is_ok()
        }));
        assert_eq!(result.served.last(), Some(&result.overall));
        // A challenger serves only once matched rounds beside the choice it
        // replaced resolved its lead, and no one round resolves one: where
        // it runs other work than the defaults it was sampled round after
        // round. A challenger that never led keeps its one round.
        for (index, record) in result.configurations.iter().enumerate().skip(1) {
            let Outcome::Measured { points, .. } = &record.outcome else {
                continue;
            };
            let rounds = points.iter().map(|point| point.samples.len()).max().unwrap();
            if result.served.contains(&record.configuration) {
                assert!(rounds > 1, "{:?}: served {index} after {rounds} round", device.backend());
            }
        }
        // The tuned configuration prepares and runs.
        let kernel =
            split_sum::native_for_device(&device, &result.overall.specialization()).unwrap();
        let value = kernel
            .call(split_sum::Args { x: &inputs[0] })
            .unwrap()
            .value;
        assert_eq!(
            read_f32(&value),
            [exact_values(n as usize).iter().sum::<f32>()]
        );

        // A survey measures and validates every configuration with every
        // sample recorded.
        let survey = split_sum::native_tune(
            &device,
            &statics(n),
            points(&inputs),
            NumericalRequirement::BitExact,
            Strategy::Survey(SurveyPlan {
                samples: 5,
                min_sample_seconds: 0.0002,
                domains: Default::default(),
            }),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        assert_eq!(survey.configurations.len(), domain);
        for record in &survey.configurations {
            match &record.outcome {
                Outcome::Measured { points, .. } => {
                    assert_eq!(survey.admitted(record), Ok(()));
                    assert!(
                        points
                            .iter()
                            .all(|point| (1..=5).contains(&point.samples.len()))
                    );
                }
                Outcome::Excluded(exclusion) => {
                    panic!("{:?}: {exclusion:?}", device.backend())
                }
            }
        }
    }
}

#[test]
fn a_search_freezes_all_positive_demand_representatives_before_candidates() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 4096u64;
    let short = f32_tensor(&device, &[n], &exact_values(n as usize));
    let long = f32_tensor(&device, &[n], &exact_values(n as usize));
    let tune = || {
        let Strategy::Search(plan) = search() else {
            unreachable!()
        };
        split_sum::native_tune(
            &device,
            &statics(n),
            vec![
                TuningPoint {
                    label: "short".into(),
                    weight: 1.0,
                    class: None,
                    cost: 1.0,
                    workload: seismic::WorkloadRotation::All,
                    rotation: vec![split_sum::Args { x: &short }],
                    initialize: None,
                    written: Default::default(),
                },
                TuningPoint {
                    label: "long".into(),
                    weight: 3.0,
                    class: None,
                    cost: 1000.0,
                    workload: seismic::WorkloadRotation::All,
                    rotation: vec![split_sum::Args { x: &long }],
                    initialize: None,
                    written: Default::default(),
                },
            ],
            NumericalRequirement::BitExact,
            Strategy::Search(plan),
        )
        .unwrap()
    };
    let weights = |result: &seismic::TuningResult| {
        result
            .points
            .iter()
            .map(|point| (point.label.clone(), point.weight))
            .collect::<Vec<_>>()
    };
    let timed = |result: &seismic::TuningResult| {
        let defaults = result
            .configurations
            .iter()
            .find(|record| record.configuration.params["PARTS"] == 1)
            .expect("the defaults were evaluated");
        match &defaults.outcome {
            Outcome::Measured { points, .. } => points
                .iter()
                .map(|point| point.point.clone())
                .collect::<Vec<_>>(),
            Outcome::Excluded(exclusion) => panic!("{exclusion:?}"),
        }
    };
    let result = tune();
    assert_eq!(timed(&result), ["short", "long"]);
    assert_eq!(
        weights(&result),
        [("short".to_owned(), 1.0), ("long".to_owned(), 3.0)]
    );
}

#[test]
fn a_measurement_begun_runs_to_its_end_and_is_recorded() {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Instant;
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 64u64;
    let first = f32_tensor(&device, &[n], &vec![1.; n as usize]);
    let second = f32_tensor(&device, &[n], &exact_values(n as usize));
    let countdown = Rc::new(Cell::new(0usize));
    let deadline = Rc::new(Cell::new(None::<Instant>));
    let counter = countdown.clone();
    let blocked_until = deadline.clone();
    let mut source = vec![TuningPoint {
        label: "interrupted-witnesses".into(),
        weight: 1.,
        class: None,
        cost: 1.,
        workload: seismic::WorkloadRotation::Selected(vec![0]),
        rotation: vec![
            split_sum::Args { x: &first },
            split_sum::Args { x: &second },
        ],
        initialize: Some(Box::new(move || {
            let remaining = counter.get();
            if remaining > 0 {
                counter.set(remaining - 1);
                if remaining == 1 {
                    let until = blocked_until.take().unwrap();
                    std::thread::sleep(
                        until.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
                    );
                }
            }
            Ok(())
        })),
        written: Default::default(),
    }];
    let mut unit = split_sum::native_search(&device, &statics(n), NumericalRequirement::BitExact).unwrap();
    unit.prepare_defaults(&mut source).unwrap();
    let Strategy::Search(plan) = search() else {
        unreachable!()
    };
    unit.plan(UnitPlan {
        settings: plan.settings,
        min_sample_seconds: 0.0002,
        start: Vec::new(),
    });
    while unit
        .next()
        .is_some_and(|work| work.stage == Stage::Discovery)
    {
        unit.perform(&mut source).unwrap();
    }
    let started = unit.standing();
    let work = unit.next().expect("a challenger is planned");
    assert_eq!(work.stage, Stage::Measure);
    // Placing the challenger restores the state once; the next reset is its
    // first numerical witness. Run that witness past the instant its budget
    // would end: the measurement begun completes and is recorded.
    let overrun_until = Instant::now() + Duration::from_millis(300);
    deadline.set(Some(overrun_until));
    countdown.set(2);
    let performed = unit.perform(&mut source).unwrap();
    assert_eq!(countdown.get(), 0, "the witness was reached");
    assert!(Instant::now() > overrun_until);
    assert_eq!(performed.stage, Stage::Measure);
    assert!(performed.actual >= Duration::from_millis(300));
    let measured = unit.standing();
    let challengers = measured.measured - started.measured;
    assert!(
        challengers >= 1,
        "work begun is never abandoned for the time it took"
    );
    assert_eq!(measured.programs, started.programs + challengers);
    // The measurement is in the unit's record at once, whatever it leads to.
    assert_eq!(unit.terminal().unwrap().configurations.len(), measured.measured);
}

/// What a candidate publishes at a point is determined by the work it runs
/// there. Configurations that differ only in a parameter no launch working
/// at the point reads share a key there: the device runs one of them for
/// observation, once per witness, and the others take that observation.
#[test]
fn a_numerical_observation_is_made_once_per_point_and_key() {
    use std::cell::Cell;
    use std::rc::Rc;
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    // The staged launch alone works at this size, whatever `SMALL` is: the
    // mirrored launch's participant count changes nothing here.
    let n = 32u64;
    let witnesses = [
        f32_tensor(&device, &[n], &exact_values(n as usize)),
        f32_tensor(&device, &[n], &vec![1.; n as usize]),
    ];
    let resets = Rc::new(Cell::new(0usize));
    let counted = resets.clone();
    let survey = gated_sum::native_tune(
        &device,
        &NativeSpecialization::new(),
        vec![TuningPoint {
            label: "staged".into(),
            weight: 1.,
            class: None,
            cost: 1.,
            workload: seismic::WorkloadRotation::All,
            rotation: witnesses
                .iter()
                .map(|x| gated_sum::Args { x })
                .collect(),
            // Called once as a configuration is placed, and once before
            // every observed invocation: the default's as the reference
            // and a candidate's.
            initialize: Some(Box::new(move || {
                counted.set(counted.get() + 1);
                Ok(())
            })),
            written: Default::default(),
        }],
        NumericalRequirement::BitExact,
        Strategy::Survey(SurveyPlan {
            samples: 1,
            min_sample_seconds: 0.0002,
            domains: Default::default(),
        }),
    )
    .unwrap();
    let default = &survey.served[0];
    let mut keys = std::collections::BTreeSet::new();
    for record in &survey.configurations {
        let Outcome::Measured { points, .. } = &record.outcome else {
            panic!("{:?}", record.outcome)
        };
        assert_eq!(survey.admitted(record), Ok(()));
        if record.configuration == *default {
            // The reference is measured against nothing.
            assert!(record.observations.is_empty());
            continue;
        }
        // Every candidate holds the observation made under its own key.
        let [observation] = record.observations[..] else {
            panic!("{:?}", record.observations)
        };
        assert_eq!(survey.observations[observation].key, points[0].key);
        assert_eq!(survey.observations[observation].rotations.len(), witnesses.len());
        keys.insert(points[0].key.clone());
    }
    let candidates = survey.configurations.len() - 1;
    assert!(
        keys.len() < candidates,
        "{candidates} candidates must share some of their {} keys for this to test reuse",
        keys.len()
    );
    assert_eq!(survey.observations.len(), keys.len());
    // The device ran each witness once for the default and once per
    // distinct key, not once per candidate.
    assert_eq!(
        resets.get(),
        survey.configurations.len() + witnesses.len() * (1 + keys.len())
    );
}

/// The census runs the native default once per performance invocation and
/// retains that run as the reference there; it observes no other witness.
/// Selecting fewer performance invocations shrinks what is placed and timed.
#[test]
fn the_census_retains_the_default_at_its_performance_invocations() {
    use std::cell::Cell;
    use std::rc::Rc;
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 64u64;
    let first = f32_tensor(&device, &[n], &vec![1.; n as usize]);
    let adversarial = f32_tensor(&device, &[n], &exact_values(n as usize));
    let census = |workload, invocations: usize| {
        let resets = Rc::new(Cell::new(0));
        let observed = resets.clone();
        let mut points = vec![TuningPoint {
            label: "two-witnesses".into(),
            weight: 1.,
            class: None,
            cost: 1.,            workload,
            rotation: vec![
                split_sum::Args { x: &first },
                split_sum::Args { x: &adversarial },
            ],
            initialize: Some(Box::new(move || {
                observed.set(observed.get() + 1);
                Ok(())
            })),
            written: Default::default(),
        }];
        let mut unit =
            split_sum::native_search(&device, &statics(n), NumericalRequirement::BitExact).unwrap();
        unit.prepare_defaults(&mut points).unwrap();
        let Strategy::Search(plan) = search() else {
            unreachable!()
        };
        unit.plan(UnitPlan {
            settings: plan.settings,
            min_sample_seconds: 0.0002,
            start: Vec::new(),
        });
        assert_eq!(unit.next().map(|work| work.stage), Some(Stage::Discovery));
        unit.perform(&mut points).unwrap();
        // Placing initializes once; the default's retained run initializes
        // before each performance invocation.
        assert_eq!(resets.get(), 1 + invocations);
        unit.terminal().unwrap()
    };
    let selected = census(seismic::WorkloadRotation::Selected(vec![0]), 1);
    let all = census(seismic::WorkloadRotation::All, 2);
    let measurement = |receipt: &seismic::TuningResult| match &receipt.configurations[0].outcome {
        Outcome::Measured { points, .. } => points[0].rotation_bytes,
        outcome => panic!("{outcome:?}"),
    };
    assert!(
        measurement(&selected) < measurement(&all),
        "selected performance calls must not retain the other witness's input/results; shared scratch is charged once"
    );
    // The default is the reference: the census qualifies nothing.
    assert!(selected.observations.is_empty() && all.observations.is_empty());
}

#[test]
fn canonical_discovery_transports_the_frozen_objective_without_remeasurement() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 4096u64;
    let x = f32_tensor(&device, &[n], &exact_values(n as usize));
    let points = || {
        [("short", 1.0), ("required", 2.0), ("long", 3.0)]
        .into_iter()
        .map(|(label, cost)| TuningPoint {
            label: label.into(),
            weight: 1.0,
            class: None,
            cost,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![split_sum::Args { x: &x }],
            initialize: None,
            written: Default::default(),
        })
        .collect::<Vec<_>>()
    };
    let timed = |result: &seismic::TuningResult| {
        let defaults = result
            .configurations
            .iter()
            .find(|record| record.configuration.params["PARTS"] == 1)
            .expect("the defaults were evaluated");
        match &defaults.outcome {
            Outcome::Measured { points, .. } => points.clone(),
            Outcome::Excluded(exclusion) => panic!("{exclusion:?}"),
        }
    };
    // Direct tuning uses every positive-demand point as a representative.
    let mut unit = split_sum::native_search(&device, &statics(n), NumericalRequirement::BitExact).unwrap();
    let mut source = points();
    let Strategy::Search(plan) = search() else {
        unreachable!()
    };
    // The mandatory work alone leaves a legal record: the canonical
    // defaults, nothing measured.
    unit.prepare_defaults(&mut source).unwrap();
    unit.plan(UnitPlan {
        settings: plan.settings,
        min_sample_seconds: plan.min_sample_seconds,
        start: plan.start,
    });
    let mandatory = unit.terminal().unwrap();
    assert!(matches!(mandatory.method, TuningMethod::Census));
    assert!(mandatory.configurations.is_empty());
    assert!(matches!(
        mandatory.disposition,
        seismic::TuningDisposition::CanonicalFallback { .. }
    ));
    // Discovery measures the defaults one point a piece, each unpriced:
    // its operations' classes are first observed by it.
    for discovered in 1..=3 {
        let work = unit.next().unwrap();
        assert_eq!(work.stage, Stage::Discovery);
        assert_eq!(work.quote.forecast(), None);
        let performed = unit.perform(&mut source).unwrap();
        assert!(!performed.promoted);
        assert_eq!(timed(&unit.terminal().unwrap()).len(), discovered);
    }
    let census = unit.terminal().unwrap();
    let censused = timed(&census);
    assert_eq!(
        censused
            .iter()
            .map(|point| point.point.as_str())
            .collect::<Vec<_>>(),
        ["short", "required", "long"]
    );
    // The search opens over exactly these measurements; it performed
    // planning only. Each later measurement is one whole challenger.
    let started = unit.standing();
    assert!(
        started.measured == 1,
        "opening transports only the discovered canonical artifact"
    );
    // The defaults against themselves: one, to the rounding of a sum over
    // points whose weights follow their measured times.
    assert!((started.cost - 1.).abs() < 1e-12, "{}", started.cost);
    // A challenger is priced in full before it starts, from this load's
    // observations of its compile, placements, witnesses and samples.
    // Its compile is part of its price, and separable from it: the piece
    // performed as it stands compiles its own challengers' programs.
    let work = unit.next().unwrap();
    assert_eq!(work.stage, Stage::Measure);
    assert!(work.gain > 0.);
    assert!(work.forming != seismic::Quote::default());
    let performed = unit.perform(&mut source).unwrap();
    assert!(performed.spent.compiling > 0.);
    let measured = unit.standing();
    let challengers = measured.measured - started.measured;
    assert!(challengers >= 1);
    assert_eq!(measured.programs, started.programs + challengers);
    let mut steps = 0;
    let mut promoted = false;
    let mut ahead = 0;
    while let Some(work) = unit.next() {
        let before = unit.standing();
        let served = unit.terminal().unwrap().overall;
        // A challenger piece's programs compile ahead of it, off the
        // device, as a consumer's worker runs them: the unit holds what
        // was formed, and the piece itself then compiles nothing.
        let formed = unit.forming().map(|forming| {
            assert!(matches!(work.stage, Stage::Form | Stage::Measure));
            let formed = std::thread::scope(|scope| scope.spawn(|| forming.run()).join())
                .expect("the compile runs to its end");
            unit.formed(formed).unwrap();
            assert!(unit.forming().is_none(), "nothing is left to compile");
            assert_eq!(unit.next().unwrap().forming, seismic::Quote::default());
            unit.standing().programs - before.programs
        });
        let performed = unit.perform(&mut source).unwrap();
        let after = unit.standing();
        if let Some(formed) = formed {
            ahead += 1;
            assert_eq!(performed.spent.compiling, 0.);
            assert_eq!(after.measured, before.measured + formed);
            assert_eq!(after.programs, before.programs + formed);
        }
        match work.stage {
            // A lead is converted before anything else is measured, and
            // converting it measures no new configuration. A conversion is
            // decided on the finalists' matched re-measurement, not on the
            // first measurements the standing cost reports: it promoted
            // exactly when the served choice changed.
            Stage::Promote | Stage::Confirm => {
                assert_eq!(after.measured, before.measured);
                assert_eq!(
                    performed.promoted,
                    unit.terminal().unwrap().overall != served
                );
                promoted |= performed.promoted;
            }
            Stage::Measure => assert!(after.measured > before.measured),
            Stage::Form => panic!("the entry declares no form parameter"),
            Stage::Discovery => panic!("discovery precedes every challenger"),
        }
        steps += 1;
        assert!(steps < 10_000, "the pieces make progress");
    }
    assert!(ahead >= 1, "a later piece compiled ahead");
    let finished = unit.standing();
    assert!(finished.measured > started.measured && finished.measured <= finished.generated);
    let searched = unit.terminal().unwrap();
    assert!(matches!(
        searched.method,
        TuningMethod::Search {
            groups: 1,
            complete: true,
            ..
        }
    ));
    // A promoted challenger is the served choice; otherwise the defaults.
    assert_eq!(
        searched.disposition == seismic::TuningDisposition::QualifiedCandidate,
        promoted
    );
    assert_eq!(searched.configurations.len(), finished.measured);
    let measured = timed(&searched);
    assert_eq!(
        measured
            .iter()
            .map(|point| point.point.as_str())
            .collect::<Vec<_>>(),
        ["short", "required", "long"]
    );
    assert_eq!(measured, censused);
}

#[test]
fn a_spent_budget_still_forms_and_serves_the_canonical_defaults() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 4096u64;
    let x = f32_tensor(&device, &[n], &exact_values(n as usize));
    let Strategy::Search(mut plan) = search() else {
        unreachable!()
    };
    plan.allowance = Duration::ZERO;
    let result = split_sum::native_tune(
        &device,
        &statics(n),
        vec![
            TuningPoint {
                label: "first".into(),
                weight: 1.0,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![split_sum::Args { x: &x }],
                initialize: None,
                written: Default::default(),
            },
            TuningPoint {
                label: "required".into(),
                weight: 1.0,
                class: None,
                cost: 2.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![split_sum::Args { x: &x }],
                initialize: None,
                written: Default::default(),
            },
        ],
        NumericalRequirement::BitExact,
        Strategy::Search(plan),
    )
    .unwrap();
    // No budget starts no optional work and fails nothing: the mandatory
    // canonical program is formed and served, its scope unmeasured.
    assert!(matches!(
        result.disposition,
        seismic::TuningDisposition::CanonicalFallback { .. }
    ));
    assert!(result.configurations.is_empty());
    assert!(result.time.forming_seconds > 0.);
    assert_eq!(result.time.building_seconds, 0.);
}

#[test]
fn a_search_choice_passes_validation_at_every_point_or_gives_way_to_the_defaults() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 4096u64;
    // Exact values sum exactly in any order; these do not, so a
    // configuration that reorders the sum fails the exact policy here.
    let inexact = (0..n)
        .map(|index| 1.0 + (index as f32) * 1.0e-3 + 1.0 / (index as f32 + 3.0))
        .collect::<Vec<_>>();
    let timed = f32_tensor(&device, &[n], &exact_values(n as usize));
    let validated = f32_tensor(&device, &[n], &inexact);
    let result = split_sum::native_tune(
        &device,
        &statics(n),
        vec![
            TuningPoint {
                label: "timed".into(),
                weight: 1.0,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![split_sum::Args { x: &timed }],
                initialize: None,
                written: Default::default(),
            },
            TuningPoint {
                label: "validated".into(),
                weight: 0.0,
                class: None,
                cost: 1000.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![split_sum::Args { x: &validated }],
                initialize: None,
                written: Default::default(),
            },
        ],
        NumericalRequirement::BitExact,
        search(),
    )
    .unwrap();
    // Only positive-demand discovery is timed; both numerical shapes remain declared.
    assert_eq!(result.points.len(), 2);
    assert!(
        result
            .configurations
            .iter()
            .all(|record| match &record.outcome {
                Outcome::Measured { points, .. } =>
                    points.iter().all(|point| point.point == "timed"),
                Outcome::Excluded(_) => true,
            })
    );
    let defaults = &result.configurations[0].configuration;
    if result.overall != *defaults {
        let chosen = result
            .configurations
            .iter()
            .find(|record| record.configuration == result.overall)
            .unwrap();
        assert!(
            chosen
                .observations
                .iter()
                .any(|observation| result.observations[*observation].point == "validated")
                && result.admitted(chosen).is_ok(),
            "a choice other than the defaults passed at every point"
        );
    }
    // The chosen configuration prepares and sums the inexact values as the
    // reference, the defaults, does.
    let kernel = split_sum::native_for_device(&device, &result.overall.specialization()).unwrap();
    let reference = split_sum::native_for_device(&device, &defaults.specialization()).unwrap();
    let sum = |kernel: &seismic::NativeKernel<split_sum::Entry>| {
        read_f32(
            &kernel
                .call(split_sum::Args { x: &validated })
                .unwrap()
                .value,
        )
    };
    assert_eq!(sum(&kernel), sum(&reference));
}

/// A graph node whose port contradicts the kernel's static dimension is
/// rejected when the graph is sealed, not when it runs.
#[test]
fn a_graph_whose_static_dimension_differs_is_rejected_at_seal() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 2).with_param("WIDTH", 32),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let input = graph.input_for(&kernel, "x", &[("N", 63)]).unwrap();
        let result = graph
            .enqueue(
                &kernel,
                split_sum::WorkflowArgs {
                    x: input.tensor().into(),
                },
            )
            .unwrap();
        graph.export(&result.value).unwrap();
        match graph.seal() {
            Err(CallError::Invocation(InvocationError::StaticDimension {
                dimension,
                expected,
                ..
            })) => {
                assert_eq!(dimension, "N");
                assert_eq!(expected, 64);
            }
            Err(other) => panic!(
                "{:?}: expected a static-dimension error, got {other}",
                device.backend()
            ),
            Ok(_) => panic!(
                "{:?}: a contradicted static dimension sealed",
                device.backend()
            ),
        }
    }
}

/// Node scratch lives in the slot workspace: two scratch-using nodes and a
/// node without scratch run correctly across repeated runs of one slot.
#[test]
fn graph_nodes_with_scratch_run_from_the_slot_workspace() {
    for device in devices() {
        let n = 1000u64;
        let kernel = split_sum::native_for_device(
            &device,
            &statics(n).with_param("PARTS", 4).with_param("WIDTH", 64),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let first_input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
        let second_input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
        let first = graph
            .enqueue(
                &kernel,
                split_sum::WorkflowArgs {
                    x: first_input.tensor().into(),
                },
            )
            .unwrap();
        let second = graph
            .enqueue(
                &kernel,
                split_sum::WorkflowArgs {
                    x: second_input.tensor().into(),
                },
            )
            .unwrap();
        graph.export(&first.value).unwrap();
        graph.export(&second.value).unwrap();
        let plan = graph.seal().unwrap();
        // Each node's four f32 partials; the second node may reuse the first's.
        assert!(plan.workspace_bytes() >= 16, "{:?}", device.backend());
        let mut slot = plan.new_slot().unwrap();
        for run in 0..3usize {
            let first_values = (0..n as usize)
                .map(|index| ((index + run) % 5) as f32 - 2.0)
                .collect::<Vec<_>>();
            let second_values = exact_values(n as usize);
            let bytes = |values: &[f32]| {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            slot.write_input(&first_input, &bytes(&first_values))
                .unwrap();
            slot.write_input(&second_input, &bytes(&second_values))
                .unwrap();
            let (outputs, completion) = slot
                .attach(plan.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&outputs.exported(&first.value).unwrap()),
                [first_values.iter().sum::<f32>()],
                "{:?} run {run}",
                device.backend()
            );
            assert_eq!(
                read_f32(&outputs.exported(&second.value).unwrap()),
                [second_values.iter().sum::<f32>()],
                "{:?} run {run}",
                device.backend()
            );
        }
    }
}

/// The factor of node `node` of a scale chain: exact in f32.
fn chain_factor(node: usize) -> f32 {
    match node {
        0 => 1.0,
        _ if node % 2 == 0 => 0.5,
        _ => 2.0,
    }
}

/// `values` as a scale chain of `nodes` nodes computes them.
fn chained(values: &[f32], nodes: usize) -> Vec<f32> {
    values
        .iter()
        .map(|value| (0..nodes).fold(*value, |value, node| value * chain_factor(node)))
        .collect()
}

fn scale_chain(
    device: &Device,
    nodes: usize,
    m: u64,
    n: u64,
) -> (
    seismic::NativeGraphPlan,
    seismic::NativePort,
    seismic::WorkflowTensor,
) {
    let kernel =
        scale_rows::native_for_device(device, &NativeSpecialization::new().with_param("ROWS", 1))
            .unwrap();
    let mut graph = device.native_graph();
    let input = graph
        .input_for(&kernel, "x", &[("M", m), ("N", n)])
        .unwrap();
    let mut value = graph
        .enqueue(
            &kernel,
            scale_rows::WorkflowArgs {
                x: input.tensor().into(),
                factor: chain_factor(0),
            },
        )
        .unwrap()
        .value;
    for node in 1..nodes {
        value = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: (&value).into(),
                    factor: chain_factor(node),
                },
            )
            .unwrap()
            .value;
    }
    graph.export(&value).unwrap();
    (graph.seal().unwrap(), input, value)
}

/// NR6: all launches of one graph submission are encoded as one unit (one
/// serial encoder), and give the bits of the per-launch-encoder form.
#[test]
fn a_graph_submission_encodes_every_node_into_one_encoder() {
    for device in devices() {
        let nodes = 8;
        let (m, n) = (3u64, 50u64);
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let values = (0..m * n)
            .map(|index| index as f32 * 0.25)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut slot = plan.new_slot().unwrap();
        let mut run = |detail: TraceDetail| {
            let trace = device.trace_submissions(detail).unwrap();
            slot.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = slot
                .attach(plan.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            let submissions = trace.collect().unwrap();
            (read_f32(&outputs.exported(&value).unwrap()), submissions)
        };
        let (serial, serial_trace) = run(TraceDetail::Submissions);
        let (separate, separate_trace) = run(TraceDetail::Launches);
        assert_eq!(serial, chained(&values, nodes), "{:?}", device.backend());
        assert_eq!(serial, separate, "{:?}", device.backend());
        for trace in [&serial_trace, &separate_trace] {
            assert_eq!(
                trace.len(),
                1,
                "{:?}: one submission per run",
                device.backend()
            );
            assert_eq!(trace[0].launches.len(), nodes);
        }
        if device.backend() != BackendName::Cpu {
            // Only the per-launch form has a timed unit (encoder) per launch.
            assert!(
                serial_trace[0]
                    .launches
                    .iter()
                    .all(|launch| launch.device.is_none())
            );
            assert!(
                separate_trace[0]
                    .launches
                    .iter()
                    .all(|launch| launch.device.is_some())
            );
        }
    }
}

/// A sequence submits its runs as one unit, in queue order: a step of an
/// entry run and five block runs chained through two alternating output
/// leases (each recycled while a queued run still reads it, as an engine
/// step does) gives the separately submitted result, and repeated steps
/// (the same storage every time) each read their own input.
#[test]
fn a_sequence_submits_chained_runs_as_one_unit_in_queue_order() {
    for device in devices() {
        let (m, n) = (3u64, 64u64);
        let blocks = 5;
        let (entry, input, entry_value) = scale_chain(&device, 1, m, n);
        let scale = scale_rows::native_for_device(
            &device,
            &NativeSpecialization::new().with_param("ROWS", 1),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let x = graph.port(Element::f32(), &[m, n]).unwrap();
        let tripled = graph
            .enqueue(
                &scale,
                scale_rows::WorkflowArgs {
                    x: x.tensor().into(),
                    factor: 3.0,
                },
            )
            .unwrap()
            .value;
        graph.export(&tripled).unwrap();
        let block = graph.seal().unwrap();
        let entry_family = NativeGraphFamily::new(&[entry.clone()]).unwrap();
        let block_family = NativeGraphFamily::new(&[block.clone()]).unwrap();
        let mut entry_slot = entry_family.new_slot(1).unwrap();
        let mut block_slot = block_family.new_slot(1).unwrap();
        let mut entry_output = Some(entry_family.new_output_slot().unwrap());
        let mut block_outputs = [
            Some(block_family.new_output_slot().unwrap()),
            Some(block_family.new_output_slot().unwrap()),
        ];
        for step in 0..3u64 {
            let values = (0..m * n)
                .map(|index| (index + step * 7) as f32 * 0.25)
                .collect::<Vec<_>>();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let trace = device.trace_submissions(TraceDetail::Submissions).unwrap();
            let mut sequence = device.native_sequence();
            let mut active = entry_slot.activate(&entry).unwrap();
            active.write_input(&input, &bytes).unwrap();
            let entry_outputs = active
                .attach(
                    entry.bindings(),
                    entry_output.take().unwrap().activate(&entry).unwrap(),
                )
                .unwrap()
                .queue(&mut sequence)
                .unwrap();
            let mut hidden = entry_outputs.exported(&entry_value).unwrap();
            let mut pending: [Option<seismic::NativeGraphOutputs>; 2] = [None, None];
            for index in 0..blocks {
                let parity = index % 2;
                let mut bindings = block.bindings();
                bindings.set(&x, &hidden).unwrap();
                let outputs = block_slot
                    .activate(&block)
                    .unwrap()
                    .attach(
                        bindings,
                        block_outputs[parity]
                            .take()
                            .unwrap()
                            .activate(&block)
                            .unwrap(),
                    )
                    .unwrap()
                    .queue(&mut sequence)
                    .unwrap();
                hidden = outputs.exported(&tripled).unwrap();
                // The previous output is still read by the run just queued.
                if let Some(previous) = pending[1 - parity].take() {
                    block_outputs[1 - parity] = Some(previous.recycle().unwrap());
                }
                pending[parity] = Some(outputs);
            }
            let completion = sequence.submit().unwrap();
            completion.wait().unwrap();
            let submissions = trace.collect().unwrap();
            assert_eq!(
                submissions.len(),
                1,
                "{:?}: one submission per step",
                device.backend()
            );
            assert_eq!(submissions[0].launches.len(), 1 + blocks);
            let expected = values
                .iter()
                .map(|value| value * 3f32.powi(blocks as i32))
                .collect::<Vec<_>>();
            assert_eq!(
                read_f32(&hidden),
                expected,
                "step {step} on {:?}",
                device.backend()
            );
            drop(hidden);
            entry_output = Some(entry_outputs.recycle().unwrap());
            for (parity, outputs) in pending.into_iter().enumerate() {
                if let Some(outputs) = outputs {
                    block_outputs[parity] = Some(outputs.recycle().unwrap());
                }
            }
        }
    }
}

/// Runs of one plan that bind the same storage (one workspace, one upload
/// region, one recycled output arena) replay the first run's formed launches
/// (on CUDA, one graph). Each still reads the input written for it.
#[test]
fn runs_rebinding_the_same_storage_read_their_own_inputs() {
    for device in devices() {
        let nodes = 6;
        let (m, n) = (4u64, 70u64);
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let family = NativeGraphFamily::new(&[plan.clone()]).unwrap();
        let mut slot = family.new_slot(1).unwrap();
        let mut output_slot = family.new_output_slot().unwrap();
        for run in 0..4u64 {
            let values = (0..m * n)
                .map(|index| (index + run * 100) as f32 * 0.5)
                .collect::<Vec<_>>();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let mut active = slot.activate(&plan).unwrap();
            active.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = active
                .attach(plan.bindings(), output_slot.activate(&plan).unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&outputs.exported(&value).unwrap()),
                chained(&values, nodes),
                "run {run} on {:?}",
                device.backend()
            );
            output_slot = outputs.recycle().unwrap();
        }
    }
}

/// NR5: a host read of an exported output issued before its run completes
/// waits for the run and returns the final bytes.
#[test]
fn a_host_read_before_completion_returns_the_final_bytes() {
    for device in devices() {
        let (m, n) = (64u64, 1024u64);
        let nodes = 64;
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let values = (0..m * n)
            .map(|index| (index % 1000) as f32)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut slot = plan.new_slot().unwrap();
        slot.write_input(&input, &bytes).unwrap();
        let (outputs, completion) = slot
            .attach(plan.bindings(), plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        let read = read_f32(&outputs.exported(&value).unwrap());
        assert!(
            completion.is_complete(),
            "{:?}: the read waited",
            device.backend()
        );
        completion.wait().unwrap();
        assert_eq!(read, chained(&values, nodes), "{:?}", device.backend());
    }
}

/// S8: a standalone call's scratch comes from the prepared kernel's
/// invocation workspace, which reports it.
#[test]
fn standalone_scratch_is_charged_to_the_invocation_workspace() {
    for device in devices() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 4).with_param("WIDTH", 32),
        )
        .unwrap();
        let before = kernel.invocation_workspace_bytes();
        let x = f32_tensor(&device, &[64], &exact_values(64));
        for _ in 0..2 {
            let value = kernel.call(split_sum::Args { x: &x }).unwrap().value;
            assert_eq!(read_f32(&value), [exact_values(64).iter().sum::<f32>()]);
            // The reserved sync range, then four f32 partials, reused by the
            // second call.
            assert_eq!(
                kernel.invocation_workspace_bytes(),
                before + SYNC_RESERVE + 16,
                "{:?}",
                device.backend()
            );
        }
    }
}

fn zeroed(device: &Device, n: u64) -> Tensor {
    f32_tensor(device, &[n], &vec![0.0; n as usize])
}

/// S6: a point binding the same `&mut` tensor for every configuration
/// without an initializer is rejected before anything runs.
#[test]
fn tuning_rejects_shared_mutable_state_without_an_initializer() {
    for device in devices() {
        let n = 256u64;
        let mut state = zeroed(&device, n);
        let x = f32_tensor(&device, &[n], &exact_values(n as usize));
        let points = vec![TuningPoint {
            label: "rows".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![accumulate::Args {
                state: &mut state,
                x: &x,
            }],
            initialize: None,
            written: Default::default(),
        }];
        match accumulate::native_tune(
            &device,
            &NativeSpecialization::new(),
            points,
            NumericalRequirement::BitExact,
            search(),
        ) {
            Err(TuneError::SharedMutableState { point, parameter }) => {
                assert_eq!(point, "rows");
                assert_eq!(parameter, "state");
            }
            Err(other) => panic!("{:?}: unexpected {other}", device.backend()),
            Ok(_) => panic!("{:?}: shared mutable state was tuned", device.backend()),
        }
    }
}

/// S6: validation compares `&mut` parameters, so a mapping parameter that
/// changes their bits is a misclassified-parameter defect; the initializer
/// makes every configuration start from the same state.
#[test]
fn tuning_validates_in_place_parameters_and_excludes_misclassified_ones() {
    for device in devices() {
        let n = 256u64;
        let mut state = zeroed(&device, n);
        let mut restore = state.clone();
        let zeros = vec![0u8; n as usize * 4];
        let initialize: TuningInitializer<'_> = Box::new(move || restore.write_from_host(&zeros));
        let x = f32_tensor(&device, &[n], &exact_values(n as usize));
        let points = vec![TuningPoint {
            label: "rows".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![accumulate::Args {
                state: &mut state,
                x: &x,
            }],
            initialize: Some(initialize),
            written: Default::default(),
        }];
        let result = accumulate::native_tune(
            &device,
            &NativeSpecialization::new(),
            points,
            NumericalRequirement::BitExact,
            // The misclassified value is hinted: it is witnessed.
            search_hinted(&[("BIAS", 1)]),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        // The default, recorded first, is the reference itself: it is measured
        // and carries no evidence against itself. A candidate is witnessed
        // before its timing samples or after rounds that leave it ahead, as
        // its unit orders the two: one that changes the state's bits never
        // serves, and is excluded wherever it is witnessed.
        assert!(result.configurations[0].observations.is_empty());
        let mut excluded = 0;
        for record in &result.configurations {
            match (record.configuration.params["BIAS"], &record.outcome) {
                (0, Outcome::Measured { .. }) => {
                    assert_eq!(result.admitted(record), Ok(()));
                }
                (1, outcome) => {
                    if !disagreeing_candidate_is_not_served(&result, record, &device) {
                        continue;
                    }
                    excluded += 1;
                    let Outcome::Excluded(Exclusion::Validation { point, detail }) = outcome else {
                        unreachable!("a witnessed candidate that disagrees is excluded")
                    };
                    assert_eq!(point, "rows");
                    assert!(detail.contains("subject i0 at point rows"), "{detail}");
                    // A rejected candidate keeps what was measured of it.
                    assert_eq!(
                        result.admitted(record).unwrap_err().exclusion(),
                        Exclusion::Validation {
                            point: point.clone(),
                            detail: detail.clone()
                        }
                    );
                }
                (bias, outcome) => panic!("{:?}: BIAS {bias} gave {outcome:?}", device.backend()),
            }
        }
        assert!(excluded >= 1, "{:?}: the hinted candidate was witnessed", device.backend());
        // Nothing serves on rounds alone: every choice promoted was
        // witnessed, and admitted.
        for served in result.served.iter().skip(1) {
            let record = result
                .configurations
                .iter()
                .find(|record| record.configuration == *served)
                .expect("a served choice was reached");
            assert!(
                !record.observations.is_empty() && result.admitted(record).is_ok(),
                "{:?}: served unwitnessed: {record:?}",
                device.backend()
            );
        }
        assert_eq!(result.overall.params["BIAS"], 0);
    }
}

/// Static bindings are checked once by `bind_static`; a run checks only the
/// bindings it adds, and still rejects unbound, mismatched and illegally
/// aliased external ports before anything is encoded.
#[test]
fn external_bindings_are_checked_at_attach() {
    for device in devices() {
        let (m, n) = (2u64, 40u64);
        let scale = scale_rows::native_for_device(
            &device,
            &NativeSpecialization::new().with_param("ROWS", 1),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let x = graph.port(Element::f32(), &[m, n]).unwrap();
        let doubled = graph
            .enqueue(
                &scale,
                scale_rows::WorkflowArgs {
                    x: x.tensor().into(),
                    factor: 2.0,
                },
            )
            .unwrap()
            .value;
        graph.export(&doubled).unwrap();
        let plan = graph.seal().unwrap();
        let values = (0..m * n).map(|index| index as f32).collect::<Vec<_>>();
        let input = f32_tensor(&device, &[m, n], &values);
        let wrong = f32_tensor(&device, &[m, n + 1], &vec![0.0; (m * (n + 1)) as usize]);
        assert!(matches!(
            plan.bind_static(&[(&x, &wrong)]),
            Err(CallError::Workflow(
                seismic::WorkflowError::NativePortMismatch { port: 0 }
            ))
        ));
        let bound = plan.bind_static(&[(&x, &input)]).unwrap();
        let mut slot = plan.new_slot().unwrap();
        for _ in 0..2 {
            let (outputs, completion) = slot
                .attach(bound.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            let expected = values.iter().map(|value| value * 2.0).collect::<Vec<_>>();
            assert_eq!(read_f32(&outputs.exported(&doubled).unwrap()), expected);
        }
        assert!(matches!(
            slot.attach(plan.bindings(), plan.new_outputs().unwrap())
                .err(),
            Some(CallError::Workflow(
                seismic::WorkflowError::NativePortUnbound { port: 0 }
            ))
        ));
        let mut bindings = plan.bindings();
        bindings.set(&x, &wrong).unwrap();
        assert!(matches!(
            slot.attach(bindings, plan.new_outputs().unwrap()).err(),
            Some(CallError::Workflow(
                seismic::WorkflowError::NativePortMismatch { port: 0 }
            ))
        ));

        // `accumulate` requires `state` and `x` to be disjoint; both are
        // external ports, so only the run's bindings can violate it.
        let accumulate = accumulate::native_for_device(
            &device,
            &NativeSpecialization::new()
                .with_param("BIAS", 0)
                .with_param("WIDTH", 32),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let mut state = graph.port(Element::f32(), &[n]).unwrap();
        let addend = graph.port(Element::f32(), &[n]).unwrap();
        graph
            .enqueue(
                &accumulate,
                accumulate::WorkflowArgs {
                    state: state.tensor_mut().into(),
                    x: addend.tensor().into(),
                },
            )
            .unwrap();
        let plan = graph.seal().unwrap();
        let mut slot = plan.new_slot().unwrap();
        let shared = f32_tensor(&device, &[n], &vec![1.0; n as usize]);
        let mut bindings = plan.bindings();
        bindings.set(&state, &shared).unwrap();
        bindings.set(&addend, &shared).unwrap();
        match slot.attach(bindings, plan.new_outputs().unwrap()).err() {
            Some(CallError::Invocation(InvocationError::IllegalAliasing { first, second })) => {
                assert_eq!([first.as_str(), second.as_str()], ["state", "x"]);
            }
            other => panic!(
                "{:?}: expected illegal aliasing, got {other:?}",
                device.backend()
            ),
        }
        let separate = f32_tensor(&device, &[n], &vec![2.0; n as usize]);
        let mut bindings = plan.bindings();
        bindings.set(&state, &shared).unwrap();
        bindings.set(&addend, &separate).unwrap();
        let (_, completion) = slot
            .attach(bindings, plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        completion.wait().unwrap();
        assert_eq!(
            read_f32(&shared),
            vec![3.0; n as usize],
            "{:?}",
            device.backend()
        );
    }
}

/// `gated_sum` by the portable body (the reference interpreter).
fn portable_gated_sum(values: &[f32]) -> f32 {
    use seismic_lang::{
        checked::{SourceFile, SourceSet, check_source},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        types::DType,
    };
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "conditional.seismic".into(),
        text: include_str!("../fixtures/conditional.seismic").into(),
    }]))
    .expect("the conditional fixture checks");
    let logical = module
        .entry(
            module.entry_named("gated_sum").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let x = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![values.len()],
        values.iter().map(|value| f64::from(*value)).collect(),
    ));
    let outcome = interpreter.run(&[Arg::Tensor(x)]).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("gated_sum portable body failed: {failure}");
    }
    let result = outcome.results().next().expect("one result");
    let OutcomeValue::Tensor(sum) = result.value() else {
        panic!("gated_sum returns a tensor")
    };
    sum.read(0).unwrap() as f32
}

fn gated(device: &Device, small: u64) -> seismic::NativeKernel<gated_sum::Entry> {
    gated_sum::native_for_device(
        device,
        &NativeSpecialization::new().with_param("SMALL", small),
    )
    .unwrap_or_else(|error| panic!("{:?}: SMALL {small}: {error}", device.backend()))
}

/// Exactly one of `gated_sum`'s two launches is active for every shape: the
/// call matches the portable body standalone and in a sealed graph, and a
/// launch-detail trace shows one device interval and one empty launch, by
/// declaration ordinal. At `N == SMALL` the inactive launch's grid would
/// divide by zero, and below `SMALL` it and the scratch size would underflow.
#[test]
fn conditional_launches_run_only_the_active_launch() {
    for device in devices() {
        // (N, SMALL, active launch ordinal). Vulkan forms group memory at
        // preparation, so SMALL = 2^20 (4 MiB staged) does not prepare there
        // (`inactive_launches_are_exempt_from_device_limits`).
        let cases = [
            (10u64, 64u64, 0usize),
            (64, 64, 0),
            (200, 64, 1),
            (200, 1_048_576, 0),
        ];
        let cases = cases
            .into_iter()
            .filter(|(_, small, _)| device.backend() != BackendName::Vulkan || *small == 64);
        for (n, small, active) in cases {
            let values = exact_values(n as usize);
            let expected = portable_gated_sum(&values);
            let kernel = gated(&device, small);
            let x = f32_tensor(&device, &[n], &values);
            let trace = device.trace_submissions(TraceDetail::Launches).unwrap();
            let value = kernel.call(gated_sum::Args { x: &x }).unwrap().value;
            let submissions = trace.collect().unwrap();
            let case = format!("{:?} N {n} SMALL {small}", device.backend());
            assert_eq!(read_f32(&value), [expected], "{case}");
            assert_eq!(submissions.len(), 1, "{case}");
            let launches = &submissions[0].launches;
            assert_eq!(
                launches
                    .iter()
                    .map(|launch| launch.launch)
                    .collect::<Vec<_>>(),
                [0, 1],
                "{case}: inactive launches keep their ordinal"
            );
            if device.backend() != BackendName::Cpu {
                assert!(launches[active].device.is_some(), "{case}");
                assert!(launches[1 - active].device.is_none(), "{case}");
            }

            let mut graph = device.native_graph();
            let input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
            let sum = graph
                .enqueue(
                    &kernel,
                    gated_sum::WorkflowArgs {
                        x: input.tensor().into(),
                    },
                )
                .unwrap()
                .value;
            graph.export(&sum).unwrap();
            let plan = graph.seal().unwrap();
            let mut slot = plan.new_slot().unwrap();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            slot.write_input(&input, &bytes).unwrap();
            let (outputs, completion) = slot
                .attach(plan.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&outputs.exported(&sum).unwrap()),
                [expected],
                "{case}: graph"
            );
        }
    }
}

/// An inactive launch is not checked against device limits: the staged
/// launch's group memory (512 KiB here) exceeds every device's limit, which
/// is an error only when that launch is active. Vulkan fixes group memory
/// when the kernel is prepared, so there the staged launch's 4 MiB region is
/// refused at preparation, whether or not a call would activate it.
#[test]
fn inactive_launches_are_exempt_from_device_limits() {
    for device in devices() {
        let n = 1u64 << 17;
        let values = exact_values(n as usize);
        let x = f32_tensor(&device, &[n], &values);
        let value = gated(&device, 64)
            .call(gated_sum::Args { x: &x })
            .unwrap()
            .value;
        // Every partial sum of these values is exact, so any order agrees
        // with the portable body's.
        assert_eq!(
            read_f32(&value),
            [values.iter().sum::<f32>()],
            "{:?}",
            device.backend()
        );
        if device.backend() == BackendName::Cpu {
            // The CPU route has no group-memory limit.
            continue;
        }
        if device.backend() == BackendName::Vulkan {
            let error = gated_sum::native_for_device(
                &device,
                &NativeSpecialization::new().with_param("SMALL", 1_048_576),
            )
            .err()
            .expect("a 4 MiB group-memory region is refused at preparation");
            assert!(
                error.to_string().contains("launch `gated_staged` needs")
                    && error.to_string().contains("the device allows"),
                "{error}"
            );
            continue;
        }
        match gated(&device, 1_048_576).call(gated_sum::Args { x: &x }) {
            Err(CallError::Execution(error)) => assert!(
                error.to_string().contains("launch `gated_staged` needs")
                    && error.to_string().contains("the device allows"),
                "{:?}: {error}",
                device.backend()
            ),
            Err(other) => panic!(
                "{:?}: expected a device-limit error, got {other}",
                device.backend()
            ),
            Ok(_) => panic!(
                "{:?}: an active launch beyond the group-memory limit ran",
                device.backend()
            ),
        }
    }
}

/// An inactive scratch buffer keeps its slot at the minimum charge, and its
/// size is not evaluated (it would underflow below `SMALL`).
#[test]
fn inactive_scratch_is_charged_the_minimum() {
    for device in devices() {
        let kernel = gated(&device, 64);
        let before = kernel.invocation_workspace_bytes();
        let call = |n: u64| {
            let values = exact_values(n as usize);
            let x = f32_tensor(&device, &[n], &values);
            let value = kernel.call(gated_sum::Args { x: &x }).unwrap().value;
            assert_eq!(
                read_f32(&value),
                [portable_gated_sum(&values)],
                "{:?} N {n}",
                device.backend()
            );
        };
        call(10);
        assert_eq!(
            kernel.invocation_workspace_bytes(),
            before + SYNC_RESERVE + 1,
            "{:?}",
            device.backend()
        );
        call(200);
        assert_eq!(
            kernel.invocation_workspace_bytes(),
            before + SYNC_RESERVE + 800,
            "{:?}",
            device.backend()
        );

        let workspace = |n: u64| {
            let mut graph = device.native_graph();
            let input = graph.input_for(&kernel, "x", &[("N", n)]).unwrap();
            let sum = graph
                .enqueue(
                    &kernel,
                    gated_sum::WorkflowArgs {
                        x: input.tensor().into(),
                    },
                )
                .unwrap()
                .value;
            graph.export(&sum).unwrap();
            graph.seal().unwrap().workspace_bytes()
        };
        assert!(workspace(10) < SYNC_RESERVE + 800, "{:?}", device.backend());
        assert!(
            workspace(200) >= SYNC_RESERVE + 800,
            "{:?}",
            device.backend()
        );
    }
}

// A reserved tensor backs only its leading committed rows. Recommitting
/// keeps them (device work bound afterwards reads them) and zero-fills the
/// rows it adds, including rows released by a shrink and backed again (CUDA
/// resizes its reserved address range in place, reusing granules).
#[test]
fn a_reserved_tensor_recommits_keeping_its_rows() {
    for device in devices() {
        let backend = device.backend();
        // 1 KiB rows: several 2 MiB CUDA granules are mapped, released and
        // mapped again.
        let (rows, n) = (8192u64, 256u64);
        let (grow, shrink, regrow) = (6000u64, 4u64, 5000u64);
        let row_values = |count: u64| {
            (0..count * n)
                .map(|index| (index % 97) as f32 * 0.5 + 1.0)
                .collect::<Vec<_>>()
        };
        let bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let with_zeros = |values: &[f32], count: u64| {
            let mut all = values.to_vec();
            all.resize((count * n) as usize, 0.0);
            all
        };
        let reserved = Tensor::reserved(&device, Element::f32(), &[rows, n], 8).unwrap();
        assert_eq!(reserved.committed_rows(), 8, "{backend:?}");
        assert_eq!(
            reserved.resizes_in_place(),
            backend == BackendName::Cuda,
            "{backend:?}"
        );
        assert!(
            reserved.read_to_host().is_err(),
            "{backend:?}: host access past the committed rows"
        );
        let first = row_values(8);
        reserved
            .slice_leading(0, 8)
            .unwrap()
            .write_from_host(&bytes(&first))
            .unwrap();

        let grown = reserved.recommitted(grow).unwrap();
        drop(reserved);
        assert_eq!(grown.committed_rows(), grow, "{backend:?}");
        assert_eq!(
            read_f32(&grown.slice_leading(0, grow).unwrap()),
            with_zeros(&first, grow),
            "{backend:?}"
        );
        grown
            .slice_leading(8, grow)
            .unwrap()
            .write_from_host(&bytes(&row_values(grow - 8)))
            .unwrap();

        let scale = scale_rows::native_for_device(
            &device,
            &NativeSpecialization::new().with_param("ROWS", 1),
        )
        .unwrap();
        let mut graph = device.native_graph();
        let x = graph.port(Element::f32(), &[8, n]).unwrap();
        let tripled = graph
            .enqueue(
                &scale,
                scale_rows::WorkflowArgs {
                    x: x.tensor().into(),
                    factor: 3.0,
                },
            )
            .unwrap()
            .value;
        graph.export(&tripled).unwrap();
        let plan = graph.seal().unwrap();
        let mut bindings = plan.bindings();
        bindings
            .set(&x, &grown.slice_leading(0, 8).unwrap())
            .unwrap();
        let (outputs, completion) = plan
            .new_slot()
            .unwrap()
            .attach(bindings, plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        completion.wait().unwrap();
        assert_eq!(
            read_f32(&outputs.exported(&tripled).unwrap()),
            first.iter().map(|value| value * 3.0).collect::<Vec<_>>(),
            "{backend:?}"
        );

        let shrunk = grown.recommitted(shrink).unwrap();
        drop(grown);
        assert_eq!(shrunk.committed_rows(), shrink, "{backend:?}");
        assert_eq!(
            shrunk.resizes_in_place(),
            backend == BackendName::Cuda,
            "{backend:?}"
        );
        let regrown = shrunk.recommitted(regrow).unwrap();
        drop(shrunk);
        assert_eq!(
            read_f32(&regrown.slice_leading(0, regrow).unwrap()),
            with_zeros(&first[..(shrink * n) as usize], regrow),
            "{backend:?}: rows released and backed again read zero"
        );
    }
}

/// A held view keeps its original physical allocation and charge. Recommit
/// must form separate backing in that case, even on CUDA VMM.
#[test]
fn a_reserved_tensor_recommits_away_from_a_held_view() {
    for device in devices() {
        let old = Tensor::reserved(&device, Element::f32(), &[128, 256], 64).unwrap();
        let held = old.slice_leading(0, 1).unwrap();
        let expected = (0..256).map(|value| value as f32).collect::<Vec<_>>();
        let encoded = expected
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        held.clone().write_from_host(&encoded).unwrap();
        let prior = old.observe_storage();
        let old_charge = prior.charged_bytes().unwrap();
        let baseline = device.memory_usage().charged;
        device.set_memory_limit(Some(baseline));
        assert!(old.recommitted(16).is_err());
        assert_eq!(device.memory_usage().charged, baseline);
        assert_eq!(prior.charged_bytes(), Some(old_charge));
        device.set_memory_limit(None);
        let new = old.recommitted(16).unwrap();
        assert!(!new.shares_allocation(&old));
        if device.backend() == BackendName::Cuda {
            assert!(new.resizes_in_place());
        }
        assert_eq!(prior.charged_bytes(), Some(old_charge));
        assert_eq!(
            device.memory_usage().charged,
            baseline + new.storage_bytes()
        );
        drop(old);
        assert_eq!(prior.charged_bytes(), Some(old_charge));
        assert_eq!(read_f32(&held), expected);
        drop(held);
        assert_eq!(prior.charged_bytes(), None);
        assert_eq!(
            device.memory_usage().charged,
            baseline + new.storage_bytes() - old_charge
        );
    }
}

#[test]
fn an_exclusive_reserved_tensor_shrinks_under_its_existing_charge() {
    for device in devices() {
        let old = Tensor::reserved(&device, Element::f32(), &[128, 256], 64).unwrap();
        let before = device.memory_usage().charged;
        let old_charge = old.storage_bytes();
        device.set_memory_limit(Some(before));
        let result = old.recommitted(16);
        device.set_memory_limit(None);
        if device.backend() == BackendName::Cuda {
            let new = result.unwrap();
            assert_eq!(new.committed_rows(), 16);
            assert_eq!(
                device.memory_usage().charged,
                before - old_charge + new.storage_bytes()
            );
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn a_failed_second_recommit_preserves_the_first_plane() {
    for device in devices() {
        let first = Tensor::reserved(&device, Element::f32(), &[128, 256], 64).unwrap();
        let second = Tensor::reserved(&device, Element::f32(), &[128, 256], 64).unwrap();
        let original = (0..256).map(|value| value as f32).collect::<Vec<_>>();
        let encoded = original
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        first
            .slice_leading(0, 1)
            .unwrap()
            .write_from_host(&encoded)
            .unwrap();
        let baseline = device.memory_usage().charged;
        let first_grown = first.recommitted(96).unwrap();
        let with_first_growth = device.memory_usage().charged;
        device.set_memory_limit(Some(with_first_growth));
        assert!(second.recommitted(96).is_err());
        device.set_memory_limit(None);
        drop(first_grown);
        assert_eq!(device.memory_usage().charged, baseline);
        assert_eq!(read_f32(&first.slice_leading(0, 1).unwrap()), original);
    }
}

/// What a search's record holds of `record`, a candidate whose numerics
/// disagree with the native default's, in whichever order its unit took a
/// challenger's two tests (its rounds beside the served choice and its
/// numerical witness: the unit orders them by what each has cost it and how
/// often each ended a challenger). It never serves. Witnessed, it is
/// excluded for validation and keeps its artifact and its one observation.
/// Not excluded, it was timed first, its rounds did not leave it ahead of
/// the served choice, and it was never witnessed: it holds no observation.
/// Whether it was witnessed.
fn disagreeing_candidate_is_not_served(
    result: &seismic::TuningResult,
    record: &seismic::ConfigurationRecord,
    device: &Device,
) -> bool {
    let backend = device.backend();
    let default = &result.configurations[0];
    assert!(
        !result.served.contains(&record.configuration) && result.overall != record.configuration,
        "{backend:?}: served {:?}",
        record.configuration
    );
    match &record.outcome {
        Outcome::Excluded(Exclusion::Validation { .. }) => {
            assert!(record.artifact.is_some() && record.artifact != default.artifact);
            assert_eq!(record.observations.len(), 1, "{backend:?}");
            assert!(result.admitted(record).is_err(), "{backend:?}");
            true
        }
        Outcome::Measured { .. } => {
            assert!(
                record.observations.is_empty(),
                "{backend:?}: witnessed and not excluded: {record:?}"
            );
            false
        }
        outcome => panic!("{backend:?}: {outcome:?}"),
    }
}

/// The native default is the reference: a candidate that disagrees with it
/// never serves, even when the default itself is wrong, and is excluded
/// wherever it is witnessed. A defect the default shares is invisible to
/// tuning; model evaluation owns it. Both other numerics are hinted, so
/// each is witnessed, and excluded. The rest of these candidates cost what
/// the default does, so one timed before its witness is not left ahead and
/// may end measured and never witnessed: which of them were is the unit's
/// order, and is not asserted.
#[test]
fn candidates_disagreeing_with_the_native_default_are_never_served() {
    use seismic_native_tests::wrong_default;
    for device in devices() {
        let mut state = f32_tensor(&device, &[4], &[2.; 4]);
        let x = f32_tensor(&device, &[4], &[1.; 4]);
        let mut reset = state.clone();
        let initial: Vec<_> = [2f32; 4]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let result = wrong_default::native_tune(
            &device,
            &NativeSpecialization::new(),
            vec![TuningPoint {
                label: "mutable".into(),
                weight: 1.,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![wrong_default::Args {
                    state: &mut state,
                    x: &x,
                }],
                initialize: Some(Box::new(move || reset.write_from_host(&initial))),
                written: Default::default(),
            }],
            NumericalRequirement::BitExact,
            search_hinted(&[("BIAS", 2), ("BIAS", 0)]),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        // Whatever serves has the default's (wrong) numerics: no
        // configuration that corrects them was served.
        assert_eq!(result.overall.params["BIAS"], 1, "{:?}", device.backend());
        let default = &result.configurations[0];
        assert_eq!(result.served[0], default.configuration);
        // A device whose only parameter that keeps the numerics is the
        // default's serves the default, unqualified.
        assert_eq!(
            matches!(
                result.disposition,
                seismic::TuningDisposition::CanonicalFallback { .. }
            ),
            result.served.len() == 1
        );
        assert!(!result.observations.is_empty());
        let keys = result
            .observations
            .iter()
            .map(|observation| (&observation.point, &observation.key))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            keys.len(),
            result.observations.len(),
            "an observation is made once per point and key"
        );
        // The default is the reference: nothing is measured of it.
        assert!(default.observations.is_empty());
        let mut disagreeing = 0;
        let mut witnessed = std::collections::BTreeMap::<u64, usize>::new();
        for record in &result.configurations {
            let bias = record.configuration.params["BIAS"];
            if bias == 1 {
                // A mapping of the default's numerics agrees with it.
                assert!(matches!(record.outcome, Outcome::Measured { .. }));
                assert_eq!(result.admitted(record), Ok(()));
            } else {
                disagreeing += 1;
                if disagreeing_candidate_is_not_served(&result, record, &device) {
                    *witnessed.entry(bias).or_default() += 1;
                }
            }
        }
        assert!(
            witnessed.contains_key(&2) && witnessed.contains_key(&0),
            "{:?}: both other numerics were witnessed and excluded: {witnessed:?}",
            device.backend()
        );
        eprintln!(
            "{:?}: {} of {disagreeing} disagreeing candidates were witnessed",
            device.backend(),
            witnessed.values().sum::<usize>()
        );
    }
}

/// Being fast gets no disagreeing candidate served. `PACE` 0 skips the
/// work the default (1) does and writes other bits: whichever of its two
/// tests the unit takes first, it is witnessed and excluded, because rounds
/// that leave a challenger ahead of the served choice are followed by its
/// witness. `PACE` 6 writes other bits too and takes six times the
/// default's work: it never serves, and whether it was witnessed is the
/// unit's order.
///
/// Nothing asserted follows from that order or from the host's timing: the
/// paces differ by whole multiples of work that takes a millisecond or so
/// (0.8 ms a pace on an M4 Pro's CPU, 2 ms on its GPU, 2026-10-10). The
/// order itself is what the unit observes, and the search is laid out so
/// that it comes to time first on its own: the paces that agree with the
/// default are hinted, so they are the unit's first session (taken witness
/// first, as every first session is) and trail there without a refusal,
/// and restoring the state takes a tenth of a second, which every witness
/// pays and no round does.
#[test]
fn a_disagreeing_candidate_ahead_in_its_rounds_is_witnessed_and_excluded() {
    use seismic_native_tests::paced_wrong_default;
    const N: usize = 64;
    for device in devices() {
        let mut state = f32_tensor(&device, &[N as u64], &[2.; N]);
        let x = f32_tensor(&device, &[N as u64], &[1.; N]);
        let mut reset = state.clone();
        let initial: Vec<_> = [2f32; N]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let Strategy::Search(mut plan) = search() else {
            unreachable!()
        };
        // One call is a millisecond or so of the kernel's own work: a
        // sample of one already measures the kernel, not its submission.
        plan.min_sample_seconds = 0.0005;
        plan.start = (2..=5)
            .map(|pace| [("PACE".to_owned(), pace)].into_iter().collect())
            .collect();
        let began = std::time::Instant::now();
        let result = paced_wrong_default::native_tune(
            &device,
            &NativeSpecialization::new(),
            vec![TuningPoint {
                label: "paced".into(),
                weight: 1.,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![paced_wrong_default::Args {
                    state: &mut state,
                    x: &x,
                }],
                initialize: Some(Box::new(move || {
                    std::thread::sleep(Duration::from_millis(100));
                    reset.write_from_host(&initial)
                })),
                written: Default::default(),
            }],
            NumericalRequirement::BitExact,
            Strategy::Search(plan),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        let default = &result.configurations[0];
        assert_eq!(default.configuration.params["PACE"], 1);
        assert_eq!(result.served[0], default.configuration);
        assert_eq!(result.overall.params["PACE"], 1, "{:?}", device.backend());
        let (mut fast, mut slow, mut slow_witnessed) = (0, 0, 0);
        for record in result.configurations.iter().skip(1) {
            match record.configuration.params["PACE"] {
                0 => {
                    fast += 1;
                    assert!(
                        disagreeing_candidate_is_not_served(&result, record, &device),
                        "{:?}: ahead in its rounds and not witnessed: {record:?}",
                        device.backend()
                    );
                }
                6 => {
                    slow += 1;
                    slow_witnessed +=
                        usize::from(disagreeing_candidate_is_not_served(&result, record, &device));
                }
                pace => {
                    // The default's numerics: measured, and admitted where
                    // witnessed. More of the default's work leads nothing.
                    assert!(
                        matches!(record.outcome, Outcome::Measured { .. }),
                        "{:?}: {record:?}",
                        device.backend()
                    );
                    assert_eq!(result.admitted(record), Ok(()));
                    assert!(pace == 1 || !result.served.contains(&record.configuration));
                }
            }
        }
        assert!(fast >= 1 && slow >= 1, "the search reached both ends");
        // A slow candidate measured and not witnessed was timed first: so
        // was the fast one of its session, excluded all the same.
        eprintln!(
            "{:?}: {fast} fast excluded; {slow_witnessed} of {slow} slow witnessed; \
             {} configurations in {:.1?}",
            device.backend(),
            result.configurations.len(),
            began.elapsed()
        );
    }
}

/// A search holds every launch group to the native default: the groups'
/// candidates disagree with these (faulty) defaults, so the defaults stand.
/// Fidelity to the source is evaluation's job, not tuning's. The fixture's
/// launches are formed separately from one Metal source.
#[test]
fn a_search_of_launch_groups_keeps_the_defaults_when_every_candidate_disagrees() {
    use seismic_native_tests::scoped_wrong_default;
    let Ok(device) = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
    else {
        return;
    };
    let small = f32_tensor(&device, &[7], &[1.; 7]);
    let large = f32_tensor(&device, &[35], &[1.; 35]);
    let points = [&small, &large]
        .into_iter()
        .enumerate()
        .map(|(i, x)| TuningPoint {
            label: format!("p{i}"),
            weight: 1.,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![scoped_wrong_default::Args { x }],
            initialize: None,
            written: Default::default(),
        })
        .collect::<Vec<_>>();
    let result = scoped_wrong_default::native_tune(
        &device,
        &NativeSpecialization::new(),
        points,
        NumericalRequirement::BitExact,
        search(),
    )
    .unwrap();
    assert_eq!(result.overall.launches[0]["ROWS"], 1);
    assert_eq!(result.overall.launches[1]["ROWS"], 4);
}

/// A unit's measurements are the same facts under every requirement: their
/// identity covers what is measured (the comparison, and the activation type
/// subjects are measured against) and no α.
#[test]
fn measurement_identity_covers_the_comparison_and_no_alpha() {
    for device in devices() {
        let x = f32_tensor(&device, &[4], &[1.; 4]);
        let mut state = f32_tensor(&device, &[4], &[0.; 4]);
        let mut run = |policy: NumericalRequirement| {
            let mut reset = state.clone();
            let bytes: Vec<_> = [0f32; 4]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            accumulate::native_tune(
                &device,
                &NativeSpecialization::new(),
                vec![TuningPoint {
                    label: "state".into(),
                    weight: 1.,
                    class: None,
                    cost: 1.0,
                    workload: seismic::WorkloadRotation::All,
                    rotation: vec![accumulate::Args {
                        state: &mut state,
                        x: &x,
                    }],
                    initialize: Some(Box::new(move || reset.write_from_host(&bytes))),
                    written: Default::default(),
                }],
                policy,
                // The changed numerics are hinted: every run witnesses them.
                search_hinted(&[("BIAS", 1)]),
            )
            .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()))
        };
        let steps = |alpha, activation| NumericalRequirement::RoundingSteps { alpha, activation };
        let exact = run(NumericalRequirement::BitExact);
        let strict = run(steps(1., None));
        let loose = run(steps(1e9, None));
        assert_eq!(exact.measurement_identity, strict.measurement_identity);
        assert_eq!(strict.measurement_identity, loose.measurement_identity);
        // A candidate measured under two requirements was measured to the
        // same numbers under both; α alone decides which of them the
        // record admits. What a search goes on to ask for follows what it
        // admitted, so the candidates of two searches need not be the
        // same: the ones both reached are compared, the changed numerics
        // one parameter from the defaults among them.
        let shared = |left: &seismic::TuningResult, right: &seismic::TuningResult| {
            let mut biased = false;
            for observation in &left.observations {
                let other = right.observations.iter().find(|other| {
                    other.point == observation.point && other.key == observation.key
                });
                if let Some(other) = other {
                    assert_eq!(observation, other, "{:?}", device.backend());
                    biased |= observation.key.values.get("BIAS") == Some(&1);
                }
            }
            biased
        };
        assert!(shared(&strict, &loose));
        assert!(shared(&exact, &strict));
        assert!(strict.rejections().count() > loose.rejections().count());
        let narrower = run(steps(1., Some(seismic::DType::BF16)));
        assert_ne!(strict.measurement_identity, narrower.measurement_identity);
    }
}

#[test]
fn rejected_candidates_stop_after_the_first_timed_invocation() {
    use seismic_native_tests::wrong_default;
    for device in devices() {
        let x = f32_tensor(&device, &[4], &[1.; 4]);
        let mut state = f32_tensor(&device, &[4], &[2.; 4]);
        let mut reset = state.clone();
        let initial: Vec<_> = [2f32; 4]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let observed = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let captures = observed.clone();
        let result = wrong_default::native_tune(
            &device,
            &NativeSpecialization::new(),
            vec![TuningPoint {
                label: "early".into(),
                weight: 1.,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![wrong_default::Args {
                    state: &mut state,
                    x: &x,
                }],
                initialize: Some(Box::new(move || {
                    captures.borrow_mut().push(read_f32(&reset)[0]);
                    reset.write_from_host(&initial)
                })),
                written: Default::default(),
            }],
            NumericalRequirement::BitExact,
            search_hinted(&[("BIAS", 2), ("BIAS", 0)]),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        // Every subsequent initialization observes the preceding call
        // before reset. The default (BIAS 1) is the reference; the
        // candidates disagreeing with it produce 4.0 (BIAS 2) and 3.0
        // (BIAS 0) in one call from the restored state. Each is hinted, so
        // it is witnessed before it is timed: it ran once, for its one
        // observation, and never again. One the search reached later and
        // timed first ran its samples' many calls from a restored state,
        // and at most its one witness after them.
        let mut values = observed.borrow().clone();
        values.push(read_f32(&state)[0]);
        for (bias, written) in [(2, 4.0), (0, 3.0)] {
            let candidates = result
                .configurations
                .iter()
                .filter(|record| record.configuration.params["BIAS"] == bias)
                .count();
            let ran = values.iter().filter(|&&value| value == written).count();
            assert!(
                (1..=candidates).contains(&ran),
                "{:?}: BIAS {bias} ran {ran} times for {candidates} candidates: {values:?}",
                device.backend()
            );
        }
        assert_eq!(result.overall.params["BIAS"], 1);
    }
}

#[test]
fn pooled_results_cannot_hide_a_candidates_missing_writes() {
    use seismic_native_tests::omitted_write;
    for device in devices() {
        let x = f32_tensor(&device, &[4], &[1., 2., 3., 4.]);
        let result = omitted_write::native_tune(
            &device,
            &NativeSpecialization::new(),
            vec![TuningPoint {
                label: "missing-write".into(),
                weight: 1.,
                class: None,
                cost: 1.0,
                workload: seismic::WorkloadRotation::All,
                rotation: vec![omitted_write::Args { x: &x }],
                initialize: None,
                written: Default::default(),
            }],
            NumericalRequirement::BitExact,
            // The missing-write candidate is hinted: it is witnessed.
            search_hinted(&[("WRITE", 0)]),
        )
        .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        assert_eq!(
            result.configurations[0].configuration.params["WRITE"], 1,
            "the good candidate must populate pooled output before the missing-write candidate"
        );
        assert_eq!(result.overall.params["WRITE"], 1);
        // Every witness of a candidate that writes nothing excludes it; one
        // the search reached after the hinted one and timed first was not
        // left ahead, and was never witnessed.
        let witnessed = result
            .configurations
            .iter()
            .filter(|record| record.configuration.params["WRITE"] == 0)
            .filter(|record| disagreeing_candidate_is_not_served(&result, record, &device))
            .count();
        assert!(witnessed >= 1, "{:?}", device.backend());
    }
}

/// Validation observes a `&mut` parameter's declared written rows, not its
/// whole storage. A point that also observes the other rows rejects a write
/// there whatever error the requirement admits in the written ones; a point
/// that declares no rows holds the whole storage to the requirement.
#[test]
fn declared_rows_scope_state_observation_and_observing_the_rest_rejects_stray_writes() {
    for device in devices() {
        stray_writes_are_observed_where_declared(device);
    }
}

fn stray_writes_are_observed_where_declared(device: Device) {
    use seismic_native_tests::stray_write;
    const N: usize = 8;
    let x = f32_tensor(&device, &[N as u64], &[1.; N]);
    let initial: Vec<u8> = [0f32; N].iter().flat_map(|v| v.to_le_bytes()).collect();
    // The second point of a run: none, one observed whole, or one that
    // declares the written rows and observes the rest.
    #[derive(Clone, Copy, PartialEq)]
    enum Guard {
        None,
        Whole,
        Rest,
    }
    let stray = |guard: Guard, requirement: NumericalRequirement| {
        let mut declared = f32_tensor(&device, &[N as u64], &[0.; N]);
        let mut observed = f32_tensor(&device, &[N as u64], &[0.; N]);
        let point = |label: &str, state: &Tensor, written: Option<bool>| {
            let mut reset = state.clone();
            let initial = initial.clone();
            (
                label.to_owned(),
                Box::new(move || reset.write_from_host(&initial)) as TuningInitializer<'static>,
                written
                    .map(|observe_rest| {
                        (
                            "state".to_owned(),
                            seismic::WrittenRows {
                                rows: 0..N as u64 - 1,
                                observe_rest,
                            },
                        )
                    })
                    .into_iter()
                    .collect(),
            )
        };
        let mut points = vec![point("declared", &declared, Some(false))];
        match guard {
            Guard::None => {}
            Guard::Whole => points.push(point("whole", &observed, None)),
            Guard::Rest => points.push(point("rest", &observed, Some(true))),
        }
        let states = [&mut declared, &mut observed];
        let result = stray_write::native_tune(
            &device,
            &NativeSpecialization::new(),
            points
                .into_iter()
                .zip(states)
                .map(|((label, initialize, written), state)| TuningPoint {
                    label,
                    weight: 1.,
                    class: None,
                    cost: 1.0,
                    workload: seismic::WorkloadRotation::All,
                    rotation: vec![stray_write::Args { state, x: &x }],
                    initialize: Some(initialize),
                    written,
                })
                .collect::<Vec<_>>(),
            requirement,
            // The stray write is hinted: it is witnessed, and it is the
            // first of its value the search reaches.
            search_hinted(&[("STRAY", 1)]),
        )
        .unwrap();
        result
            .configurations
            .into_iter()
            .find(|record| record.configuration.params["STRAY"] == 1)
            .unwrap()
            .outcome
    };
    let exact = NumericalRequirement::BitExact;
    // Any finite error in a written row is within this.
    let anything = NumericalRequirement::RoundingSteps {
        alpha: f64::MAX,
        activation: None,
    };
    assert!(
        matches!(stray(Guard::None, exact.clone()), Outcome::Measured { .. }),
        "rows outside the declared ones are not observed"
    );
    assert!(
        matches!(
            stray(Guard::Whole, exact.clone()),
            Outcome::Excluded(Exclusion::Validation { .. })
        ),
        "a point observed whole rejects the stray write"
    );
    assert!(
        matches!(stray(Guard::Whole, anything.clone()), Outcome::Measured { .. }),
        "a point that declares no rows holds every row to the requirement alone"
    );
    for requirement in [exact, anything] {
        match stray(Guard::Rest, requirement) {
            Outcome::Excluded(Exclusion::Validation { point, detail }) => {
                assert_eq!(point, "rest");
                assert!(
                    detail.contains("subject i0:unwritten")
                        && detail.contains("1 elements differ where none may"),
                    "{detail}"
                );
            }
            outcome => panic!("a write outside the written rows was admitted: {outcome:?}"),
        }
    }
}

/// A point whose build the device's memory limit refuses is a piece that
/// did not fit its room, not a failure of the unit: the piece ends with
/// nothing measured, the point is priced at what its build demanded, and
/// once that is free the same discovery runs.
#[test]
fn a_build_the_memory_limit_refuses_prices_its_point_and_is_offered_again() {
    use seismic::{PointInputs, PointSource, PointSpec};

    /// Points whose inputs are allocated on the device as they are built,
    /// each with state of `held` bytes the source keeps beside them.
    struct Allocated<'d> {
        device: &'d Device,
        n: u64,
        held: [u64; 2],
        kept: [Option<Tensor>; 2],
    }
    impl PointSource<'static, split_sum::Entry> for Allocated<'_> {
        fn release(&mut self, point: usize) {
            self.kept[point] = None;
        }
        fn points(&self) -> Vec<PointSpec> {
            (0..2)
                .map(|point| PointSpec {
                    label: format!("p{point}"),
                    weight: 1.,
                    class: None,
                    cost: (point + 1) as f64,
                    census: true,
                })
                .collect()
        }
        fn build(&mut self, point: usize) -> Result<PointInputs<'static>, String> {
            let state = Tensor::zeros(self.device, Element::f32(), &[self.held[point] / 4])
                .map_err(|error| error.to_string())?;
            let x = f32_tensor(self.device, &[self.n], &exact_values(self.n as usize));
            self.kept[point] = Some(state);
            Ok(seismic::point_inputs::<split_sum::Entry>(
                vec![split_sum::Args { x: &x }],
                seismic::WorkloadRotation::All,
                None,
                Default::default(),
            ))
        }
    }

    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 64;
    let large = 8u64 << 20;
    let mut source = Allocated {
        device: &device,
        n,
        held: [16, large],
        kept: [None, None],
    };
    let mut unit =
        split_sum::native_search(&device, &statics(n), NumericalRequirement::BitExact).unwrap();
    unit.prepare_defaults(&mut source).unwrap();
    let Strategy::Search(plan) = search() else {
        unreachable!()
    };
    unit.plan(UnitPlan {
        settings: plan.settings,
        min_sample_seconds: 0.0002,
        start: Vec::new(),
    });
    assert_eq!(unit.next().unwrap().stage, Stage::Discovery);
    unit.perform(&mut source).unwrap();
    // The second point's state is refused by the device's limit.
    let charged = device.memory_usage().charged;
    device.set_memory_limit(Some(charged + large / 2));
    let work = unit.next().unwrap();
    assert_eq!(work.stage, Stage::Discovery);
    assert_eq!(work.footprint.unpriced, 1, "an unbuilt point has no price");
    let performed = unit
        .perform(&mut source)
        .expect("a refused build is not the unit's failure");
    assert_eq!(performed.stage, Stage::Discovery);
    assert!(!performed.promoted);
    let work = unit.next().unwrap();
    assert_eq!(
        work.stage,
        Stage::Discovery,
        "the point is still undiscovered"
    );
    assert_eq!(work.footprint.unpriced, 0);
    assert!(
        work.footprint.priced >= large,
        "the point is priced at what its build demanded: {}",
        work.footprint.priced
    );
    // With room for it, the same point is discovered.
    device.set_memory_limit(None);
    unit.perform(&mut source).unwrap();
    assert_ne!(unit.next().map(|work| work.stage), Some(Stage::Discovery));
}

/// A discovery whose inputs fit but whose measurement of the defaults the
/// device's enforced memory limit refuses is not the unit's failure either:
/// the point is priced at what the piece demanded and offered again, as a
/// refused build is.
#[test]
fn a_defaults_measurement_the_memory_limit_refuses_prices_its_point_and_is_offered_again() {
    use seismic::{PointInputs, PointSource, PointSpec};

    /// Points whose inputs are allocated on the device as they are built;
    /// building `clamp` sets the device's limit to what is then charged, so
    /// that only the measurement's own allocations are refused.
    struct Clamping<'d> {
        device: &'d Device,
        n: u64,
        clamp: Option<usize>,
        kept: [Option<Tensor>; 2],
    }
    impl PointSource<'static, split_sum::Entry> for Clamping<'_> {
        fn release(&mut self, point: usize) {
            self.kept[point] = None;
        }
        fn points(&self) -> Vec<PointSpec> {
            (0..2)
                .map(|point| PointSpec {
                    label: format!("p{point}"),
                    weight: 1.,
                    class: None,
                    cost: (point + 1) as f64,
                    census: true,
                })
                .collect()
        }
        fn build(&mut self, point: usize) -> Result<PointInputs<'static>, String> {
            let x = f32_tensor(self.device, &[self.n], &exact_values(self.n as usize));
            self.kept[point] = Some(x.clone());
            if self.clamp == Some(point) {
                self.device
                    .set_memory_limit(Some(self.device.memory_usage().charged));
            }
            Ok(seismic::point_inputs::<split_sum::Entry>(
                vec![split_sum::Args { x: &x }],
                seismic::WorkloadRotation::All,
                None,
                Default::default(),
            ))
        }
    }

    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 64;
    let mut source = Clamping {
        device: &device,
        n,
        clamp: None,
        kept: [None, None],
    };
    let mut unit =
        split_sum::native_search(&device, &statics(n), NumericalRequirement::BitExact).unwrap();
    unit.prepare_defaults(&mut source).unwrap();
    let Strategy::Search(plan) = search() else {
        unreachable!()
    };
    unit.plan(UnitPlan {
        settings: plan.settings,
        min_sample_seconds: 0.0002,
        start: Vec::new(),
    });
    assert_eq!(unit.next().unwrap().stage, Stage::Discovery);
    unit.perform(&mut source).unwrap();
    source.clamp = Some(1);
    let work = unit.next().unwrap();
    assert_eq!(work.stage, Stage::Discovery);
    let performed = unit
        .perform(&mut source)
        .expect("a refused measurement of the defaults is not the unit's failure");
    assert_eq!(performed.stage, Stage::Discovery);
    let work = unit.next().unwrap();
    assert_eq!(
        work.stage,
        Stage::Discovery,
        "the point is still undiscovered"
    );
    assert_eq!(work.footprint.unpriced, 0);
    assert!(work.footprint.priced > 0);
    // With room for it, the same point is discovered.
    source.clamp = None;
    device.set_memory_limit(None);
    unit.perform(&mut source).unwrap();
    assert_ne!(unit.next().map(|work| work.stage), Some(Stage::Discovery));
}
