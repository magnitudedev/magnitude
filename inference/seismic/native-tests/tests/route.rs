//! The direct-native route on every backend this host can open: static
//! dimensions, tuning parameters, `where`, scratch, several launches, shared
//! memory, conditional (`when`) launches and scratch, graphs with
//! host-written inputs, asynchronous submission, measurement and tuning.

use seismic::measure::{measure, Measure, Stop};
use seismic::{
    BackendName, CallError, Device, Devices, Element, Exclusion, ExecutionError,
    InvocationError, NativeGraphFamily, NativeSpecialization, NumericalRequirement, Outcome,
    RegionError, Placing, ReservationCharges, ReservedContract, Resizer, SearchPlan, SearchSettings, Stage, Strategy, SurveyPlan, Tensor,
    TensorError,
    TraceDetail, TuneError, TuningInitializer, TuningMethod, TuningPoint, UnitPlan,
};
use seismic_native_tests::{
    accumulate, gated_sum, marked_groups, scale_records, scale_rows, scoped_scale, split_sum,
    written_marks,
};
use std::num::NonZeroUsize;
use std::time::Duration;

/// The range every non-empty scratch arena reserves at its start for `sync`
/// scratch (the runtime's `SYNC_SCRATCH_BYTES`).
const SYNC_RESERVE: u64 = 64 * 1024;

/// A search whose time outlasts its convergence on the test entries.
fn search() -> Strategy {
    Strategy::Search(SearchPlan {
        allowance: Duration::from_secs(3600),

        settings: SearchSettings { error_rate: 0.05 },
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

/// What `device` charges on each store it reaches: its own ledgers.
fn storage_charged(device: &Device) -> std::collections::BTreeMap<seismic::StoreId, u64> {
    device.ledgers().own.iter().map(|(store, view)| (*store, view.charged)).collect()
}

/// The store a buffer placed as `placing` is placed on first.
fn store_of(device: &Device, placing: Placing) -> seismic::StoreId {
    let path = device.info().paths(placing)[0];
    device.topology().access_path(path).expect("a placement's path belongs to its topology").store()
}

/// Each store has been charged, since `before`, exactly what `charges`
/// places there: the backing in its reserved kind's storage's store, a
/// record table in its placement's.
fn assert_charged_where_placed(
    device: &Device,
    before: &std::collections::BTreeMap<seismic::StoreId, u64>,
    charges: ReservationCharges,
) {
    let mut expected = before.clone();
    *expected.entry(store_of(device, Placing::Reserved(charges.contract))).or_default() += charges.backing;
    if let Some(table) = charges.table {
        *expected.entry(store_of(device, Placing::Fixed(table.placement()))).or_default() += table.bytes();
    }
    let backend = device.backend();
    assert_eq!(storage_charged(device), expected, "{backend:?}: each store charges only the owners placed there");
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
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
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

/// A launch with `grid` dispatches the groups its grid tensor holds when it
/// runs, up to its declared bound: none of an empty grid, some of a smaller
/// one.
#[test]
fn metal_launch_dispatches_the_grid_it_reads() {
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
        return;
    };
    let kernel = marked_groups::native_for_device(&device, &NativeSpecialization::new())
        .expect("Metal has the grid fixture");
    let words = |values: &[i32]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    for groups in [0i32, 3, 8] {
        let grid = Tensor::from_host(&device, Element::i32(), &[3], &words(&[groups, 1, 1])).unwrap();
        let mut marks = Tensor::from_host(&device, Element::i32(), &[8], &words(&[7; 8])).unwrap();
        kernel
            .call(marked_groups::Args {
                grid: &grid,
                marks: &mut marks,
            })
            .expect("grid call");
        let expected = (0..8).map(|group| i32::from(group < groups)).collect::<Vec<_>>();
        let marked = marks
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(marked, expected, "grid of {groups} groups");
    }
}

/// On every backend, a launch dispatches the grid its call's first launch
/// wrote into scratch, up to its declared bound: none of an empty grid,
/// every group of the bound, and each group at its own place on both axes.
/// On CUDA the groups run as a grid-stride loop of the device's resident
/// blocks, so the bound's 4096 groups outnumber them.
#[test]
fn a_launch_dispatches_the_grid_its_call_wrote() {
    let words = |values: &[i32]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    for device in seismic_native_tests::every_device() {
        let kernel = written_marks::native_for_device(&device, &NativeSpecialization::new())
            .expect("every backend has the written-grid fixture");
        for (n, count) in [(8u64, 0i32), (8, 3), (8, 8), (4096, 4093)] {
            let count_tensor = Tensor::from_host(&device, Element::i32(), &[1], &words(&[count])).unwrap();
            let mut marks =
                Tensor::from_host(&device, Element::i32(), &[2, n], &words(&vec![7; 2 * n as usize])).unwrap();
            kernel
                .call(written_marks::Args {
                    count: &count_tensor,
                    marks: &mut marks,
                })
                .expect("written-grid call");
            let marked = marks
                .read_to_host()
                .unwrap()
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            let expected = (0..2 * n).map(|index| i32::from(((index % n) as i32) < count)).collect::<Vec<_>>();
            assert_eq!(marked, expected, "{:?}: {count} of {n} groups", device.backend());
        }
    }
}

/// A slab view's binding is its address table, not its elements, so a
/// launch grid binds none: refused at a standalone call's arguments and at
/// a native graph's port, before anything is encoded.
#[test]
fn metal_launch_grid_binds_no_slab_view() {
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
        return;
    };
    let kernel = marked_groups::native_for_device(&device, &NativeSpecialization::new())
        .expect("Metal has the grid fixture");
    let mut slabs = seismic::SlabTensor::new(
        &device,
        3,
        3,
        vec![seismic::SlabRegion {
            element: Element::i32(),
            row_shape: vec![],
        }],
    )
    .unwrap();
    slabs.add_slab().unwrap();
    let grid = slabs.logical_region(0).unwrap();
    let mut marks = Tensor::from_host(&device, Element::i32(), &[8], &[0; 32]).unwrap();
    match kernel.call(marked_groups::Args {
        grid: &grid,
        marks: &mut marks,
    }) {
        Err(CallError::Invocation(InvocationError::SlabGrid { parameter })) => assert_eq!(parameter, "grid"),
        other => panic!("a slab grid at a call: {:?}", other.map(|_| ())),
    }

    let mut graph = device.native_graph();
    let port = graph.port(Element::i32(), &[3]).unwrap();
    let mut marked = graph.port(Element::i32(), &[8]).unwrap();
    graph
        .enqueue(
            &kernel,
            marked_groups::WorkflowArgs {
                grid: port.tensor().into(),
                marks: marked.tensor_mut().into(),
            },
        )
        .unwrap();
    let plan = graph.seal().unwrap();
    let mut bindings = plan.bindings();
    assert_eq!(
        bindings.set(&port, &grid),
        Err(seismic::WorkflowError::NativePortSlabGrid { port: 0 })
    );
}

/// Serving prepares the configuration tuning chose as tuning formed it: no
/// program is formed a second time.
#[test]
fn serving_the_tuned_choice_forms_nothing_again() {
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
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
    let formed = device.programs_formed().programs;
    scoped_scale::native_for_device(&device, &result.overall.specialization())
        .expect("the chosen configuration prepares");
    assert_eq!(device.programs_formed().programs, formed);
}

#[test]
fn metal_scoped_tuning_searches_launches_separately() {
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
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
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
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
            discovered = Some(device.device_ledger().charged);
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
            let performed = unit.perform(&mut source, None).unwrap();
            assert_eq!(performed.spent.compiling, 0.);
            ahead += 1;
            // The piece's high-water mark is no less than either end.
            assert!(performed.memory.peak_bytes >= performed.memory.ended_bytes);
            assert!(performed.memory.peak_bytes >= performed.memory.began_bytes);
            performed.memory.ended_bytes
        } else {
            assert_eq!(work.forming, seismic::Quote::default());
            unit.perform(&mut source, None).unwrap().memory.ended_bytes
        };
        // What a piece holds is the device's charges and the process's
        // host bytes: no less than the charges.
        let charged = device.device_ledger().charged;
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
    assert!(
        device.device_ledger().charged < discovered,
        "{held:?}"
    );
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
    let devices = Devices::discover(&[]).expect("device discovery");
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
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
    for device in seismic_native_tests::every_device() {
        let n = 1000usize;
        let values = exact_values(n);
        let expected: f32 = values.iter().sum();
        let x = f32_tensor(&device, &[n as u64], &values);
        let implementation = split_sum::native_implementation(&device)
            .expect("bundle")
            .expect("split_sum has an implementation on every backend");
        // The declared domain, with what a CPU device adds of its own (each
        // launch's participant count and the tier) at its defaults.
        let configurations = implementation
            .admissible(&implementation.implementation().with_owned_defaults(statics(n as u64)))
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
    for device in seismic_native_tests::every_device() {
        let implementation = split_sum::native_implementation(&device).unwrap().unwrap();
        let admissible = implementation
            .admissible(&implementation.implementation().with_owned_defaults(statics(2)))
            .unwrap();
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
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
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
                    factor: seismic::Argument::from(2.0),
                },
            )
            .unwrap();
        let second = graph
            .enqueue(
                &kernel,
                scale_rows::WorkflowArgs {
                    x: (&first.value).into(),
                    factor: seismic::Argument::from(3.0),
                },
            )
            .unwrap();
        graph.export(&second.value).unwrap();
        let plan = graph.seal().unwrap();
        assert!(plan.upload_bytes() >= m * n * 4);
        let family = NativeGraphFamily::new(&[plan.clone()], 1).unwrap();
        // One upload region per run kept in flight below.
        let mut slot = family.new_slot(&[family.upload_bytes(); 3]).unwrap();
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
    for device in seismic_native_tests::every_device() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(4096).with_param("PARTS", 4).with_param("WIDTH", 64),
        )
        .unwrap();
        let rotation = (0..3)
            .map(|_| f32_tensor(&device, &[4096], &exact_values(4096)))
            .collect::<Vec<_>>();
        let point = kernel
            .point(rotation.iter().map(|x| split_sum::Args { x }).collect())
            .unwrap();
        assert!(point.rotation_bytes() >= 3 * 4096 * 4);
        let measurement = measure(
            point,
            &Measure {
                stop: Stop::Count(NonZeroUsize::new(3).unwrap()),
                min_sample_seconds: 0.0005,
            },
            None,
        )
        .unwrap();
        assert_eq!(measurement.samples.len(), 3);
        assert!(measurement.median > 0.0, "{:?}", device.backend());
    }
}

#[test]
fn tuning_searches_from_the_defaults_and_validates_its_choice() {
    for device in seismic_native_tests::every_device() {
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
            let rounds = points
                .iter()
                .map(|point| point.samples.len())
                .max()
                .unwrap();
            if result.served.contains(&record.configuration) {
                assert!(
                    rounds > 1,
                    "{:?}: served {index} after {rounds} round",
                    device.backend()
                );
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
    let device = Devices::discover(&[])
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
fn already_expired_work_keeps_an_unmeasured_default_without_running_witnesses() {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Instant;
    let device = Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let n = 64u64;
    let x = f32_tensor(&device, &[n], &vec![1.; n as usize]);
    let resets = Rc::new(Cell::new(0));
    let counted = resets.clone();
    let mut source = vec![TuningPoint {
        label: "expired-before-start".into(),
        weight: 1.,
        class: None,
        cost: 1.,
        workload: seismic::WorkloadRotation::All,
        rotation: vec![split_sum::Args { x: &x }],
        initialize: Some(Box::new(move || {
            counted.set(counted.get() + 1);
            Ok(())
        })),
        written: Default::default(),
    }];
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
    let before = unit.terminal().unwrap();
    let performed = unit.perform(&mut source, Some(Instant::now())).unwrap();
    assert!(performed.interrupted);
    assert!(!performed.promoted);
    assert_eq!(resets.get(), 0);
    assert!(unit.next().is_none());
    let after = unit.terminal().unwrap();
    assert_eq!(after.overall, before.overall);
    assert!(after.configurations.is_empty());
    assert!(after.observations.is_empty());
}

#[test]
fn expiry_during_reset_preserves_the_choice_without_rejecting_the_candidate() {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Instant;
    let device = Devices::discover(&[])
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
    while unit
        .next()
        .is_some_and(|work| work.stage == Stage::Discovery)
    {
        unit.perform(&mut source, None).unwrap();
    }
    let started = unit.standing();
    let work = unit.next().expect("a challenger is planned");
    assert_eq!(work.stage, Stage::Measure);
    // Placing the challenger restores the state once; the next reset is its
    // first numerical witness. Its reset crosses the cutoff: the
    // subsequent reference/candidate submission must not start.
    let overrun_until = Instant::now() + Duration::from_millis(300);
    deadline.set(Some(overrun_until));
    countdown.set(2);
    let before = unit.terminal().unwrap();
    let performed = unit.perform(&mut source, Some(overrun_until)).unwrap();
    assert_eq!(countdown.get(), 0, "the witness was reached");
    assert!(Instant::now() > overrun_until);
    assert_eq!(performed.stage, Stage::Measure);
    assert!(performed.interrupted);
    assert!(!performed.promoted);
    assert!(performed.actual >= Duration::from_millis(300));
    let measured = unit.standing();
    assert_eq!(measured.measured, started.measured);
    assert!(unit.next().is_none());
    let after = unit.terminal().unwrap();
    assert_eq!(after.overall, before.overall);
    assert_eq!(after.configurations, before.configurations);
    assert!(
        after.observations.is_empty(),
        "an incomplete witness is not evidence"
    );
}

/// What a candidate publishes at a point is determined by the work it runs
/// there. Configurations that differ only in a parameter no launch working
/// at the point reads share a key there: the device runs one of them for
/// observation, once per witness, and the others take that observation. A
/// survey records such configurations only where another point tells them
/// apart: one whose moved parameter no launch it runs at any point reads
/// runs the default's work everywhere and is not recorded.
#[test]
fn a_numerical_observation_is_made_once_per_point_and_key() {
    use std::cell::Cell;
    use std::rc::Rc;
    let device = Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    // At 32 the staged launch alone works, whatever `SMALL` is: the mirrored
    // launch's participant count changes nothing there. At 200 the mirrored
    // launch works where `SMALL` is 64, so that count is surveyed there.
    let sizes = [32u64, 200];
    let witnesses = sizes.map(|n| {
        [
            f32_tensor(&device, &[n], &exact_values(n as usize)),
            f32_tensor(&device, &[n], &vec![1.; n as usize]),
        ]
    });
    let resets = Rc::new(Cell::new(0usize));
    let points = sizes
        .iter()
        .zip(&witnesses)
        .map(|(n, witnesses)| {
            let counted = resets.clone();
            TuningPoint {
                label: format!("n{n}"),
                weight: 1.,
                class: None,
                cost: 1.,
                workload: seismic::WorkloadRotation::All,
                rotation: witnesses.iter().map(|x| gated_sum::Args { x }).collect(),
                // Called once as a configuration is placed, and once before
                // every observed invocation: the default's as the reference
                // and a candidate's.
                initialize: Some(Box::new(move || {
                    counted.set(counted.get() + 1);
                    Ok(())
                })),
                written: Default::default(),
            }
        })
        .collect::<Vec<_>>();
    let survey = gated_sum::native_tune(
        &device,
        &NativeSpecialization::new(),
        points,
        NumericalRequirement::BitExact,
        Strategy::Survey(SurveyPlan {
            samples: 1,
            min_sample_seconds: 0.0002,
            domains: Default::default(),
        }),
    )
    .unwrap();
    let default = &survey.served[0];
    let default_workers = |record: &seismic::ConfigurationRecord| {
        record.configuration.params["cpu.workers.1"] == default.params["cpu.workers.1"]
    };
    // Where the mirrored launch never works (`SMALL` past both sizes), its
    // participant count is recorded at its default alone.
    assert!(
        survey
            .configurations
            .iter()
            .filter(|record| record.configuration.params["SMALL"] != 64)
            .all(default_workers)
    );
    let mut keys = sizes.map(|_| std::collections::BTreeSet::new());
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
        // Every candidate holds the observation made under its own key at
        // each point.
        assert_eq!(record.observations.len(), sizes.len(), "{:?}", record.observations);
        for (point, (measured, keys)) in points.iter().zip(&mut keys).enumerate() {
            let observation = record
                .observations
                .iter()
                .map(|&index| &survey.observations[index])
                .find(|observation| observation.point == format!("n{}", sizes[point]))
                .unwrap();
            assert_eq!(observation.key, measured.key);
            assert_eq!(observation.rotations.len(), witnesses[point].len());
            keys.insert(measured.key.clone());
        }
    }
    let candidates = survey.configurations.len() - 1;
    assert!(
        keys[0].len() < candidates,
        "{candidates} candidates must share some of their {} keys at 32 for this to test reuse",
        keys[0].len()
    );
    let distinct = keys.iter().map(|keys| keys.len()).sum::<usize>();
    assert_eq!(survey.observations.len(), distinct);
    // The device ran each witness once for the default and once per
    // distinct key at each point, not once per candidate.
    assert_eq!(
        resets.get(),
        sizes.len() * survey.configurations.len() + 2 * (sizes.len() + distinct)
    );
}

/// The census runs the native default once per performance invocation and
/// retains that run as the reference there; it observes no other witness.
/// Selecting fewer performance invocations shrinks what is placed and timed.
#[test]
fn the_census_retains_the_default_at_its_performance_invocations() {
    use std::cell::Cell;
    use std::rc::Rc;
    let device = Devices::discover(&[])
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
            cost: 1.,
            workload,
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
        unit.perform(&mut points, None).unwrap();
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
    let device = Devices::discover(&[])
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
    let mut unit =
        split_sum::native_search(&device, &statics(n), NumericalRequirement::BitExact).unwrap();
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
        let performed = unit.perform(&mut source, None).unwrap();
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
    // observations of its placements, witnesses and samples. A CPU unit
    // compiles nothing (its kernels are in the binary): a piece prepares
    // its challenger as it measures it, so it quotes no compile and has
    // none to run ahead, and it measures one challenger (`compiled_ahead`
    // and `replan` in tune.rs; Metal's compiles ahead are
    // `metal_compiles_run_ahead_on_another_thread_and_the_piece_then_
    // compiles_nothing`).
    let mut steps = 0;
    let mut promoted = false;
    while let Some(work) = unit.next() {
        let before = unit.standing();
        let served = unit.terminal().unwrap().overall;
        assert_eq!(work.forming, seismic::Quote::default());
        assert!(unit.forming().is_none());
        if steps == 0 {
            assert_eq!(work.stage, Stage::Measure);
            assert!(work.opportunity > 0.);
        }
        let performed = unit.perform(&mut source, None).unwrap();
        let after = unit.standing();
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
            Stage::Measure => {
                assert_eq!(after.measured, before.measured + 1);
                assert_eq!(after.programs, before.programs + 1);
            }
            Stage::Form => panic!("the entry declares no form parameter"),
            Stage::Discovery => panic!("discovery precedes every challenger"),
        }
        steps += 1;
        assert!(steps < 10_000, "the pieces make progress");
    }
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
    let device = Devices::discover(&[])
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
    let device = Devices::discover(&[])
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
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
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
                factor: seismic::Argument::from(chain_factor(0)),
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
                    factor: seismic::Argument::from(chain_factor(node)),
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
    for device in seismic_native_tests::every_device() {
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
        // Only the per-launch form has a timed unit (encoder, pool job) per
        // launch.
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

/// A sequence submits its runs as one unit, in queue order: a step of an
/// entry run and five block runs chained through two alternating output
/// leases (each recycled while a queued run still reads it, as an engine
/// step does) gives the separately submitted result, and repeated steps
/// (the same storage every time) each read their own input.
#[test]
fn a_sequence_submits_chained_runs_as_one_unit_in_queue_order() {
    for device in seismic_native_tests::every_device() {
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
                    factor: seismic::Argument::from(3.0),
                },
            )
            .unwrap()
            .value;
        graph.export(&tripled).unwrap();
        let block = graph.seal().unwrap();
        let entry_family = NativeGraphFamily::new(&[entry.clone()], 1).unwrap();
        // A launch on the block slot runs the block `blocks` times in one
        // submission, each run replaying a graph of the block's shape.
        let block_family = NativeGraphFamily::with_launches(&[block.clone()], &[vec![block.clone(); blocks]], 1, &[]).unwrap();
        let mut entry_slot = entry_family.new_slot(&[entry_family.upload_bytes()]).unwrap();
        let mut block_slot = block_family.new_slot(&[block_family.upload_bytes()]).unwrap();
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
            let mut sequence = device.native_sequence(seismic::NativeQueue::Compute);
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
    for device in seismic_native_tests::every_device() {
        let nodes = 6;
        let (m, n) = (4u64, 70u64);
        let (plan, input, value) = scale_chain(&device, nodes, m, n);
        let family = NativeGraphFamily::new(&[plan.clone()], 1).unwrap();
        let mut slot = family.new_slot(&[family.upload_bytes()]).unwrap();
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
    for device in seismic_native_tests::every_device() {
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

/// The device storage one standalone call charges while it runs beyond its
/// results: the most the device's charge rises over the call, less its
/// results' bytes. The call's scratch is its own, so the charge is back
/// where it was once the call and its results are gone. Read from the
/// charge, not from the least bound the call runs under: a backend that
/// carves allocations from blocks it already holds (CUDA) admits a call
/// inside that slack under any bound, as it takes no device memory more.
fn standalone_scratch_bytes(
    device: &Device,
    call: impl Fn() -> Result<Tensor, CallError>,
) -> u64 {
    let before = device.device_ledger().charged;
    device.restart_high_water();
    let results = call().unwrap().storage_bytes();
    let charged = device.device_ledger();
    assert_eq!(charged.charged, before);
    charged.high_water - before - results
}

/// S8: a standalone call's scratch is its own, released when the call ends;
/// the prepared kernel holds only its scalar-result slots.
#[test]
fn standalone_scratch_is_the_calls_own() {
    for device in seismic_native_tests::every_device() {
        let kernel = split_sum::native_for_device(
            &device,
            &statics(64).with_param("PARTS", 4).with_param("WIDTH", 32),
        )
        .unwrap();
        let before = kernel.scalar_result_bytes();
        let x = f32_tensor(&device, &[64], &exact_values(64));
        let call = || {
            kernel
                .call(split_sum::Args { x: &x })
                .map(|results| results.value)
        };
        assert_eq!(
            read_f32(&call().unwrap()),
            [exact_values(64).iter().sum::<f32>()]
        );
        // The reserved sync range, then four f32 partials.
        assert_eq!(
            standalone_scratch_bytes(&device, call),
            SYNC_RESERVE + 16,
            "{:?}",
            device.backend()
        );
        assert_eq!(
            kernel.scalar_result_bytes(),
            before,
            "{:?}",
            device.backend()
        );
    }
}

fn zeroed(device: &Device, n: u64) -> Tensor {
    f32_tensor(device, &[n], &vec![0.0; n as usize])
}

/// S6: a point binding the same `&mut` tensor for every configuration
/// without an initializer is rejected before anything runs.
#[test]
fn tuning_rejects_shared_mutable_state_without_an_initializer() {
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
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
        assert!(
            excluded >= 1,
            "{:?}: the hinted candidate was witnessed",
            device.backend()
        );
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

/// Static bindings are checked once by `bind_static`, which refuses a
/// disjoint pair of them that aliases; a run checks only the bindings it
/// adds, and still rejects unbound, mismatched and illegally aliased
/// external ports before anything is encoded.
#[test]
fn external_bindings_are_checked_at_attach() {
    for device in seismic_native_tests::every_device() {
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
                    factor: seismic::Argument::from(2.0),
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
                seismic::WorkflowError::NativePortBinding { port: 0, .. }
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
                seismic::WorkflowError::NativePortBinding { port: 0, .. }
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
        // Both ports bound statically: the aliasing is refused once, when
        // the plan is bound, so no run of it is ever formed.
        match plan.bind_static(&[(&state, &shared), (&addend, &shared)]).err() {
            Some(CallError::Invocation(InvocationError::IllegalAliasing { first, second })) => {
                assert_eq!([first.as_str(), second.as_str()], ["state", "x"]);
            }
            other => panic!(
                "{:?}: expected illegal aliasing at static binding, got {other:?}",
                device.backend()
            ),
        }
        // One port static, the other a run's: the run's attach refuses it.
        let bound = plan.bind_static(&[(&state, &shared)]).unwrap();
        let mut bindings = bound.bindings();
        bindings.set(&addend, &shared).unwrap();
        assert!(matches!(
            slot.attach(bindings, plan.new_outputs().unwrap()).err(),
            Some(CallError::Invocation(InvocationError::IllegalAliasing { .. }))
        ));
    }
}

/// `gated_sum` by the portable body (the reference interpreter).
fn portable_gated_sum(values: &[f32]) -> f32 {
    use seismic_lang::{
        checked::{SourceFile, SourceSet},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        types::DType,
    };
    // Its native declarations' sources are read beside it.
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let fixtures = std::path::Path::new(&manifest).join("fixtures");
    let module = seismic_lang::source::check_beside(
        SourceSet::new(vec![SourceFile {
            path: "conditional.seismic".into(),
            text: include_str!("../fixtures/conditional.seismic").into(),
        }]),
        &fixtures,
    )
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
    for device in seismic_native_tests::every_device() {
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
            assert!(launches[active].device.is_some(), "{case}");
            assert!(launches[1 - active].device.is_none(), "{case}");

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
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
        let kernel = gated(&device, 64);
        let scratch = |n: u64| {
            let values = exact_values(n as usize);
            let x = f32_tensor(&device, &[n], &values);
            let call = || kernel.call(gated_sum::Args { x: &x }).map(|results| results.value);
            assert_eq!(
                read_f32(&call().unwrap()),
                [portable_gated_sum(&values)],
                "{:?} N {n}",
                device.backend()
            );
            standalone_scratch_bytes(&device, call)
        };
        assert_eq!(scratch(10), SYNC_RESERVE + 1, "{:?}", device.backend());
        assert_eq!(scratch(200), SYNC_RESERVE + 800, "{:?}", device.backend());

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

/// Every opened backend whose topology reports device reservations. Refusal
/// is checked against that capability, independently of the backend name.
fn reserving_devices() -> Vec<Device> {
    seismic_native_tests::every_device().into_iter().filter(|device| {
        let reserving = device.info().reserving(ReservedContract::DeviceOnly).map(|reservation| reservation.is_some()).map_err(Clone::clone);
        let probe = Tensor::reserved(device, Element::f32(), &[2, 4], 1);
        let supported = reserving == Ok(true);
        if supported {
            drop(probe.expect("advertised device reservation must work"));
        } else {
            match reserving {
                Ok(_) => assert!(matches!(probe, Err(TensorError::Unreservable { why: None, .. })), "unadvertised reservation must be refused"),
                Err(why) => assert!(
                    matches!(&probe, Err(TensorError::Unreservable { why: Some(refused), .. }) if *refused == why),
                    "a refused reservation must be refused with its reason"
                ),
            }
            eprintln!("{:?}: device reservations unsupported", device.backend());
            assert!(std::env::var_os("SEISMIC_TEST_BACKEND").is_none(), "selected backend cannot execute reservation checks");
        }
        supported
    }).collect()
}

/// A resize to `rows` that must be refused: why, and the resizer back.
fn refused(resizer: Resizer, rows: u64) -> (TensorError, Resizer) {
    match resizer.recommit(rows) {
        Ok(_) => panic!("a resize to {rows} rows was not refused"),
        Err(refusal) => (refusal.error, refusal.resizer),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A sealed graph statically bound to a view of a reserved tensor's rows,
/// tripling them: the binding the recommit tests run unchanged across
/// recommits. It reaches them a row at a time, as chunked storage binds
/// (`scale_records`' record port on chunked backends).
fn tripling_graph(
    device: &Device,
    view: &Tensor,
) -> (
    seismic::NativeGraphPlan,
    seismic::BoundNativeGraphPlan,
    seismic::WorkflowTensor,
) {
    let scale = scale_records::native_for_device(
        device,
        &NativeSpecialization::new().with_param("ROWS", 1),
    )
    .unwrap();
    let mut graph = device.native_graph();
    let x = graph.port(Element::f32(), view.extents()).unwrap();
    let tripled = graph
        .enqueue(
            &scale,
            scale_records::WorkflowArgs {
                x: x.tensor().into(),
                factor: seismic::Argument::from(3.0),
            },
        )
        .unwrap()
        .value;
    graph.export(&tripled).unwrap();
    let plan = graph.seal().unwrap();
    let bound = plan.bind_static(&[(&x, view)]).unwrap();
    (plan, bound, tripled)
}

/// A reserved tensor recommits in place beneath a graph bound to it: the
/// tensor, its views and the bound graph stay valid with no rebind, and the
/// graph reads the rows' current contents after a grow and a shrink. Kept
/// rows keep their contents; rows backed again read zero, including those a
/// shrink left inside a kept granule. Host access past the committed rows is
/// refused.
#[test]
fn a_reserved_tensor_recommits_in_place_beneath_a_bound_graph() {
    for device in reserving_devices() {
        let backend = device.backend();
        // 1 KiB rows: 2 MiB granules (Metal, CUDA) are mapped, released and
        // mapped again.
        let (rows, n, bound) = (8192u64, 256u64, 8u64);
        let (grow, shrink, regrow) = (6000u64, 16u64, 5000u64);
        let row_values = |count: u64, salt: u64| {
            (0..count * n)
                .map(|index| ((index + salt) % 97) as f32 * 0.5 + 1.0)
                .collect::<Vec<_>>()
        };
        let with_zeros = |values: &[f32], count: u64| {
            let mut all = values.to_vec();
            all.resize((count * n) as usize, 0.0);
            all
        };
        let tripled = |values: &[f32]| values.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], bound).unwrap();
        assert_eq!(reserved.committed_rows(), bound, "{backend:?}");
        assert!(
            reserved.read_to_host().is_err(),
            "{backend:?}: host access past the committed rows"
        );
        let mut view = reserved.slice_leading(0, bound).unwrap();
        let first = row_values(bound, 0);
        view.write_from_host(&f32_bytes(&first)).unwrap();
        let (plan, graph, output) = tripling_graph(&device, &view);
        let mut slot = plan.new_slot().unwrap();
        let mut run = |expected: &[f32], step: &str| {
            let (outputs, completion) = slot
                .attach(graph.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            assert_eq!(
                read_f32(&outputs.exported(&output).unwrap()),
                tripled(expected),
                "{backend:?}: the bound graph after {step}"
            );
        };
        run(&first, "binding");

        let resizer = resizer.recommit(grow).unwrap();
        assert_eq!(reserved.committed_rows(), grow, "{backend:?}");
        assert_eq!(view.committed_rows(), bound, "{backend:?}");
        assert_eq!(
            read_f32(&reserved.slice_leading(0, grow).unwrap()),
            with_zeros(&first, grow),
            "{backend:?}: grown rows read zero"
        );
        let second = row_values(grow, 1);
        reserved
            .slice_leading(0, grow)
            .unwrap()
            .write_from_host(&f32_bytes(&second))
            .unwrap();
        run(&second[..(bound * n) as usize], "a grow");

        let resizer = resizer.recommit(shrink).unwrap();
        assert_eq!(reserved.committed_rows(), shrink, "{backend:?}");
        assert!(
            reserved.slice_leading(0, shrink + 1).unwrap().read_to_host().is_err(),
            "{backend:?}: host access past the committed rows after a shrink"
        );
        run(&second[..(bound * n) as usize], "a shrink");

        let resizer = resizer.recommit(regrow).unwrap();
        assert_eq!(
            read_f32(&reserved.slice_leading(0, regrow).unwrap()),
            with_zeros(&second[..(shrink * n) as usize], regrow),
            "{backend:?}: rows released and backed again read zero"
        );
        run(&second[..(bound * n) as usize], "a regrow");
        // A whole commitment is host-accessible as one tensor.
        resizer.recommit(rows).unwrap();
        assert_eq!(read_f32(&reserved).len(), (rows * n) as usize, "{backend:?}");
    }
}

/// A reserved tensor is charged for the granules that back its committed
/// rows, and on chunked storage for its record table too (one 8-byte device
/// address a row, formed with the tensor), and the charge follows each
/// recommit: each storage placement carries its declared owners, a
/// granule-aligned commitment is charged exactly, and any other is rounded
/// up to less than one granule more.
#[test]
fn a_reserved_tensor_is_charged_for_its_backed_granules() {
    for device in reserving_devices() {
        let backend = device.backend();
        // 1 KiB rows: 2048 rows fill a 2 MiB granule.
        let (rows, n, row) = (16384u64, 256u64, 1024u64);
        let reservation = device.info().reserving(ReservedContract::DeviceOnly).expect("it says how it reserves").expect("it reserves");
        let record = std::num::NonZeroU64::new(row).unwrap();
        let granule = reservation.of_records(seismic::ReservedRecords::of(record)).granule;
        let table = reservation.table_bytes(rows).unwrap();
        let before = storage_charged(&device);
        let (reserved, mut resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], 8).unwrap();
        let storage = reserved.observe_storage();
        for committed in [8, 2048, 6000, 4096, 1, 16384] {
            resizer = resizer.recommit(committed).unwrap();
            let charged = storage.charged_bytes().unwrap();
            let backed = charged - table;
            let bytes = committed * row;
            assert_eq!(reserved.storage_bytes(), bytes, "{backend:?}");
            assert_charged_where_placed(&device, &before, reservation.charges(ReservedContract::DeviceOnly, seismic::ReservedRecords::of(record), rows, bytes).unwrap());
            assert!(
                backed >= bytes && backed - bytes < granule,
                "{backend:?}: {committed} rows backed by {backed} charged bytes"
            );
            if bytes % granule == 0 {
                assert_eq!(backed, bytes, "{backend:?}");
            }
            assert_eq!(charged, reservation.charge(seismic::ReservedRecords::of(record), rows, bytes).unwrap(), "{backend:?}: topology charge");
        }
        drop((reserved, resizer));
        assert_eq!(storage.charged_bytes(), None);
        assert_eq!(storage_charged(&device), before, "{backend:?}: every store releases its charge");
    }
}

/// A recommit waits for submitted device work that uses the tensor: once it
/// returns, every run queued before it has completed.
#[test]
fn a_recommit_waits_for_device_work_in_flight() {
    for device in reserving_devices() {
        let backend = device.backend();
        let (rows, n, bound) = (8192u64, 256u64, 4096u64);
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], bound).unwrap();
        let view = reserved.slice_leading(0, bound).unwrap();
        let (plan, graph, _) = tripling_graph(&device, &view);
        let completions = (0..32)
            .map(|_| {
                plan.new_slot()
                    .unwrap()
                    .attach(graph.bindings(), plan.new_outputs().unwrap())
                    .unwrap()
                    .submit()
                    .unwrap()
                    .1
            })
            .collect::<Vec<_>>();
        resizer.recommit(bound + 1).unwrap();
        assert!(
            completions.iter().all(|completion| completion.is_complete()),
            "{backend:?}: a recommit returned before the work using the tensor completed"
        );
        for completion in completions {
            completion.wait().unwrap();
        }
    }
}

/// A growth the device's limit refuses leaves the tensor as it was: its
/// commitment, contents and charge. A shrink needs no grant, so it succeeds
/// at the limit.
#[test]
fn a_refused_growth_leaves_the_reserved_tensor_unchanged() {
    for device in reserving_devices() {
        let backend = device.backend();
        let (rows, n) = (16384u64, 256u64);
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], 64).unwrap();
        let contents = (0..64 * n).map(|value| value as f32).collect::<Vec<_>>();
        reserved
            .slice_leading(0, 64)
            .unwrap()
            .write_from_host(&f32_bytes(&contents))
            .unwrap();
        let storage = reserved.observe_storage();
        let charge = storage.charged_bytes().unwrap();
        let limit = device.device_ledger().charged;
        device.bound(seismic::Ledger::Own(device.device_store()), Some(limit)).unwrap();
        let (error, resizer) = refused(resizer, rows);
        assert!(
            matches!(
                error,
                TensorError::Execution(ExecutionError::AllocationCapacity { .. })
            ),
            "{backend:?}"
        );
        assert_eq!(reserved.committed_rows(), 64, "{backend:?}");
        assert_eq!(storage.charged_bytes(), Some(charge), "{backend:?}");
        assert_eq!(device.device_ledger().charged, limit);
        assert_eq!(
            read_f32(&reserved.slice_leading(0, 64).unwrap()),
            contents,
            "{backend:?}"
        );
        let resizer = resizer.recommit(16).unwrap();
        assert_eq!(reserved.committed_rows(), 16, "{backend:?}");
        device.bound(seismic::Ledger::Own(device.device_store()), None).unwrap();
        resizer.recommit(rows).unwrap();
        assert_eq!(
            read_f32(&reserved.slice_leading(0, 16).unwrap()),
            contents[..(16 * n) as usize],
            "{backend:?}"
        );
    }
}

/// Where a device binds a reserved range as one buffer once every granule
/// is backed (Metal), a change between that form and chunks holds the next
/// form's buffers beside the view's until its release, which the device
/// counts: the resize is charged them for that window, so the ledger never
/// passes its bound. A growth to every granule whose bound leaves no room
/// for them stays in chunks, which bind nothing directly; a shrink from the
/// one buffer at no headroom forms its kept chunks in its place.
#[test]
fn a_reserved_tensor_takes_its_whole_form_only_within_its_grant() {
    for device in reserving_devices() {
        let backend = device.backend();
        let reservation = device.info().reserving(ReservedContract::DeviceOnly).expect("it says how it reserves").expect("it reserves");
        if !reservation.chunks.is_some_and(|chunks| chunks.whole) {
            continue;
        }
        // 256 KiB rows, a whole number of them a granule.
        let (rows, n) = (64u64, 1u64 << 16);
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], rows / 2).unwrap();
        let store = seismic::Ledger::Own(device.device_store());
        // Room for the growth's own bytes, none for the chunks it would hold
        // beside one buffer over the range.
        let limit = device.device_ledger().charged + rows / 2 * n * 4;
        device.bound(store, Some(limit)).unwrap();
        let resizer = resizer.recommit(rows).unwrap();
        assert_eq!(reserved.committed_rows(), rows, "{backend:?}");
        assert!(!reserved.bindable(), "{backend:?}: a growth with no room beside its chunks stays in chunks");
        assert!(device.device_ledger().charged <= limit, "{backend:?}");
        device.bound(store, None).unwrap();
        let resizer = resizer.recommit(rows - 8).unwrap().recommit(rows).unwrap();
        assert!(reserved.bindable(), "{backend:?}: a growth with room takes the whole form");
        let charged = device.device_ledger().charged;
        device.bound(store, Some(charged)).unwrap();
        let resizer = resizer.recommit(rows / 2).unwrap();
        assert_eq!(reserved.committed_rows(), rows / 2, "{backend:?}");
        assert!(!reserved.bindable(), "{backend:?}");
        assert!(device.device_ledger().charged <= charged, "{backend:?}: a shrink in place is charged nothing beside");
        device.bound(store, None).unwrap();
        drop((reserved, resizer));
    }
}

/// The device's single-allocation limit bounds each native allocation
/// backing a reserved tensor, never its address range: a reservation past
/// the limit with a small commitment is made. Where one allocation backs the
/// committed bytes, a growth past the limit is refused, leaving the tensor
/// as it was; where each chunk is its own allocation (chunked storage), the
/// limit bounds a chunk, so a reservation of records whose chunk passes it
/// is refused. Devices whose limit is the host's address space (the CPU
/// backend) reserve no range that large.
#[test]
fn the_allocation_limit_bounds_each_allocation_backing_a_reservation() {
    let capacity = |error: TensorError, limit: u64| {
        matches!(
            error,
            TensorError::Execution(ExecutionError::AllocationCapacity { available, .. })
                if available == limit
        )
    };
    for device in reserving_devices() {
        let backend = device.backend();
        let limit = device.info().limits().max_allocation.value;
        if limit > 1 << 40 {
            eprintln!("{backend:?}: allocation-limit exercise requires a representable test reservation; limit={limit}");
            assert!(std::env::var_os("SEISMIC_TEST_BACKEND").is_none(), "selected allocation limit cannot be exercised");
            continue;
        }
        // 1 MiB rows, two more than the limit holds.
        let n = 1u64 << 18;
        let rows = limit / (4 * n) + 2;
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], 1).unwrap();
        assert_eq!(reserved.committed_rows(), 1, "{backend:?}");
        if device.info().reserving(ReservedContract::DeviceOnly).expect("it says how it reserves").expect("it reserves").chunked() {
            // One record more than the limit.
            match Tensor::reserved(&device, Element::f32(), &[2, limit / 4 + 1], 1) {
                Ok(_) => panic!("{backend:?}: a chunk past the limit was reserved"),
                Err(error) => assert!(capacity(error, limit), "{backend:?}"),
            }
            continue;
        }
        let (error, resizer) = refused(resizer, rows);
        assert!(capacity(error, limit), "{backend:?}");
        assert_eq!(reserved.committed_rows(), 1, "{backend:?}");
        resizer.recommit(2).unwrap();
        assert_eq!(reserved.committed_rows(), 2, "{backend:?}");
    }
}

/// A region of a reserved tensor lies within its committed rows and pins
/// them: a recommit beneath any live view of the region is refused, and once
/// the last one drops, the rows can be released.
#[test]
fn a_region_pins_a_reserved_tensors_committed_rows() {
    for device in reserving_devices() {
        let backend = device.backend();
        let (rows, n) = (8192u64, 256u64);
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], 4096).unwrap();
        assert!(
            matches!(
                reserved.region(Element::f32(), &[4097, n], 0),
                Err(TensorError::Region(RegionError::Unbacked { .. }))
            ),
            "{backend:?}: a region past the committed rows"
        );
        let region = reserved.region(Element::f32(), &[1024, n], 1024 * n * 4).unwrap();
        let view = region.slice_leading(0, 16).unwrap();
        drop(region);
        let (error, resizer) = refused(resizer, 2047);
        assert!(
            matches!(error, TensorError::Region(RegionError::Pinned { .. })),
            "{backend:?}: a shrink beneath a live view of a region"
        );
        assert_eq!(reserved.committed_rows(), 4096, "{backend:?}");
        let resizer = resizer.recommit(2048).unwrap();
        let resizer = resizer.recommit(rows).unwrap();
        drop(view);
        resizer.recommit(16).unwrap();
        assert_eq!(reserved.committed_rows(), 16, "{backend:?}");
    }
}

/// A resize is prepared beside device work and installed between
/// submissions. While a growth is prepared on another thread, runs bound to
/// the tensor read its rows unchanged, and until it is installed nothing the
/// tensor shows changes: its rows, its charge, host access past them. A
/// prepare abandoned leaves the tensor as it was. An install charges its
/// growth at once; a shrink's charge leaves when it is released, on any
/// thread, once the device work submitted before its install completes,
/// which hands the resizer back. A region pinned after a shrink's
/// prepare refuses its install, handing the resizer back.
#[test]
fn a_recommit_is_prepared_beside_device_work_and_installed_between_submissions() {
    for device in reserving_devices() {
        let backend = device.backend();
        let (rows, n, bound, grow, shrink) = (8192u64, 256u64, 1024u64, 6000u64, 16u64);
        let (reserved, resizer) = Tensor::reserved(&device, Element::f32(), &[rows, n], bound).unwrap();
        let first = (0..bound * n).map(|value| (value % 89) as f32).collect::<Vec<_>>();
        let mut view = reserved.slice_leading(0, bound).unwrap();
        view.write_from_host(&f32_bytes(&first)).unwrap();
        let storage = reserved.observe_storage();
        let charged = storage.charged_bytes().unwrap();
        let (plan, graph, output) = tripling_graph(&device, &view);
        let tripled = first.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        let run = || {
            let (outputs, completion) = plan
                .new_slot()
                .unwrap()
                .attach(graph.bindings(), plan.new_outputs().unwrap())
                .unwrap()
                .submit()
                .unwrap();
            completion.wait().unwrap();
            read_f32(&outputs.exported(&output).unwrap())
        };

        let prepared = std::thread::scope(|scope| {
            let preparing = scope.spawn(move || resizer.prepare(grow).unwrap());
            loop {
                assert_eq!(run(), tripled, "{backend:?}: a run while a growth is prepared");
                if preparing.is_finished() {
                    break preparing.join().unwrap();
                }
            }
        });
        assert_eq!(reserved.committed_rows(), bound, "{backend:?}");
        assert_eq!(
            storage.charged_bytes(),
            Some(charged),
            "{backend:?}: a prepared growth is charged once installed"
        );
        assert!(
            reserved.slice_leading(0, bound + 1).unwrap().read_to_host().is_err(),
            "{backend:?}: host access past the rows before the install"
        );
        assert_eq!(run(), tripled, "{backend:?}");
        let release = prepared.install().unwrap();
        assert_eq!(reserved.committed_rows(), grow, "{backend:?}");
        let grown = storage.charged_bytes().unwrap();
        assert!(grown >= grow * n * 4, "{backend:?}: an installed growth is charged");
        let resizer = release.release();
        let mut with_zeros = first.clone();
        with_zeros.resize((grow * n) as usize, 0.0);
        assert_eq!(
            read_f32(&reserved.slice_leading(0, grow).unwrap()),
            with_zeros,
            "{backend:?}: grown rows read zero"
        );
        assert_eq!(run(), tripled, "{backend:?}: a run after the growth");

        let resizer = resizer.prepare(shrink).unwrap().abandon();
        assert_eq!(reserved.committed_rows(), grow, "{backend:?}");
        assert_eq!(storage.charged_bytes(), Some(grown), "{backend:?}");
        assert_eq!(
            read_f32(&reserved.slice_leading(0, grow).unwrap()),
            with_zeros,
            "{backend:?}: an abandoned shrink changes nothing"
        );

        let prepared = resizer.prepare(shrink).unwrap();
        let region = reserved.region(Element::f32(), &[bound, n], 0).unwrap();
        let resizer = match prepared.install() {
            Err(refusal) => {
                assert!(
                    matches!(refusal.error, TensorError::Region(RegionError::Pinned { .. })),
                    "{backend:?}: {:?}",
                    refusal.error
                );
                refusal.resizer
            }
            Ok(_) => panic!("{backend:?}: a shrink beneath a region pinned since its prepare"),
        };
        drop(region);
        assert_eq!(reserved.committed_rows(), grow, "{backend:?}");

        // A run submitted before the install reads rows the shrink releases.
        let (_outputs, in_flight) = plan
            .new_slot()
            .unwrap()
            .attach(graph.bindings(), plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        let release = resizer.prepare(shrink).unwrap().install().unwrap();
        assert_eq!(reserved.committed_rows(), shrink, "{backend:?}");
        assert_eq!(
            storage.charged_bytes(),
            Some(grown),
            "{backend:?}: a shrink's charge leaves when it is released"
        );
        let resizer = std::thread::spawn(move || release.release()).join().unwrap();
        assert!(
            in_flight.is_complete(),
            "{backend:?}: a release frees rows only once the work submitted before its install completes"
        );
        assert!(
            storage.charged_bytes().unwrap() < grown,
            "{backend:?}: a release frees the shrink's charge"
        );
        assert_eq!(
            read_f32(&reserved.slice_leading(0, shrink).unwrap()),
            first[..(shrink * n) as usize],
            "{backend:?}: kept rows survive"
        );
        resizer.recommit(rows).unwrap();
        assert_eq!(reserved.committed_rows(), rows, "{backend:?}");
    }
}

/// A region of a reserved tensor is host-addressed exactly where its backend
/// declares its reservations host-addressed (`RESERVATIONS_HOST_ADDRESSED`,
/// which discovery reports: Metal, the CPU), and bytes the
/// host writes there are the tensor's; the tensor itself, whose rows a
/// recommit may release, is not host-addressed.
#[test]
fn a_reserved_region_is_host_addressed_where_reservations_are_host_memory() {
    for device in reserving_devices() {
        let backend = device.backend();
        let host = device.info().reserving(ReservedContract::DeviceOnly).expect("it says how it reserves").expect("it reserves").host_addressed;
        let (reserved, _resizer) = Tensor::reserved(&device, Element::f32(), &[8, 1024], 4).unwrap();
        assert_eq!(reserved.shared_address(), None, "{backend:?}");
        let region = reserved.region(Element::f32(), &[2, 1024], 2 * 1024 * 4).unwrap();
        assert_eq!(region.shared_address().is_some(), host, "{backend:?}");
        if host {
            let bytes = (0..2 * 1024 * 4).map(|index| index as u8).collect::<Vec<_>>();
            unsafe { region.write_shared(0, &bytes) }.unwrap();
            assert_eq!(region.read_to_host().unwrap(), bytes, "{backend:?}");
        }
    }
}

/// A reserved host tensor (host RAM the device reads in place) exists
/// exactly where its backend reports a host reservation, which is
/// host-addressed: it grows, the host writes a region of its rows in place,
/// a graph bound to that region reads them on the device, and it shrinks,
/// its charge following each resize: its backing on host storage, and a
/// record table (where its backend binds one) on the storage it is placed
/// in, which is never host storage.
#[test]
fn a_reserved_host_tensor_is_written_in_place_and_read_on_the_device() {
    for device in seismic_native_tests::every_device() {
        let backend = device.backend();
        let reservation = device.info().reserving(ReservedContract::Shared).expect("the device says how it reserves");
        let (rows, n) = (8192u64, 256u64);
        // The charges before the tensor is reserved.
        let before = storage_charged(&device);
        let reserved = Tensor::reserved_in(ReservedContract::Shared, &device, Element::f32(), &[rows, n], 8, None);
        let Some(reservation) = reservation else {
            assert!(
                matches!(reserved, Err(TensorError::Unreservable { .. })),
                "{backend:?}: no host reservation, none reserved"
            );
            eprintln!("{backend:?}: host reservations unsupported");
            assert!(std::env::var_os("SEISMIC_TEST_BACKEND").is_none(), "selected backend cannot execute host reservation check");
            continue;
        };
        assert!(reservation.host_addressed, "{backend:?}: a host reservation is host-addressed");
        let (reserved, resizer) = reserved.unwrap();
        let record = std::num::NonZeroU64::new(n * 4).unwrap();
        // The tensor holds its backed granules and its record table, each
        // charged to the storage it is placed in.
        let charged = |committed: u64, resize: &str| {
            let bytes = committed * n * 4;
            assert_eq!(reserved.charged_bytes(), reservation.charge(seismic::ReservedRecords::of(record), rows, bytes).unwrap(), "{backend:?}: {resize}");
            assert_charged_where_placed(&device, &before, reservation.charges(ReservedContract::Shared, seismic::ReservedRecords::of(record), rows, bytes).unwrap());
        };
        charged(8, "its backed granules");
        // 1 KiB rows: 2 MiB granules are mapped, then released.
        let grown = 6000u64;
        let resizer = resizer.recommit(grown).unwrap();
        charged(grown, "a growth's granules");
        // A region far into the growth, written through its host address.
        let (first, count) = (5000u64, 16u64);
        let region = reserved.region(Element::f32(), &[count, n], first * n * 4).unwrap();
        let address = region.shared_address().expect("a pinned region of host memory");
        let values = (0..count * n).map(|index| (index % 89) as f32 + 0.25).collect::<Vec<_>>();
        let bytes = f32_bytes(&values);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), address.as_ptr(), bytes.len()) };
        let (plan, graph, output) = tripling_graph(&device, &region);
        let (outputs, completion) = plan
            .new_slot()
            .unwrap()
            .attach(graph.bindings(), plan.new_outputs().unwrap())
            .unwrap()
            .submit()
            .unwrap();
        completion.wait().unwrap();
        let tripled = values.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        assert_eq!(
            read_f32(&outputs.exported(&output).unwrap()),
            tripled,
            "{backend:?}: the device reads what the host wrote in place"
        );
        drop((outputs, graph, plan, region));
        let _resizer = resizer.recommit(16).unwrap();
        assert_eq!(reserved.committed_rows(), 16, "{backend:?}");
        charged(16, "a shrink releases its granules");
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
    for device in seismic_native_tests::every_device() {
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
/// paces differ by whole multiples of a chain of dependent steps no
/// compiler shortens (`fixtures/metal/paced.metal`), far more than a call's
/// submission on every backend, and the device times them. The
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
    for device in seismic_native_tests::every_device() {
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
        // A call is far more of the kernel's own work than its submission:
        // a sample of one already measures the kernel.
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
                    slow_witnessed += usize::from(disagreeing_candidate_is_not_served(
                        &result, record, &device,
                    ));
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
    let Ok(device) = Devices::discover(&[])
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

/// A unit's measurements are the same facts under every requirement's α:
/// α judges them and changes none.
#[test]
fn measurements_are_the_same_facts_under_every_alpha() {
    for device in seismic_native_tests::every_device() {
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
        // A candidate measured under two requirements was measured to the
        // same numbers under both; α alone decides which of them the
        // record admits. What a search goes on to ask for follows what it
        // admitted, so the candidates of two searches need not be the
        // same: the ones both reached are compared, the changed numerics
        // one parameter from the defaults among them.
        let shared = |left: &seismic::TuningResult, right: &seismic::TuningResult| {
            let mut biased = false;
            for observation in &left.observations {
                let other = right
                    .observations
                    .iter()
                    .find(|other| other.point == observation.point && other.key == observation.key);
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
    }
}

#[test]
fn rejected_candidates_stop_after_the_first_timed_invocation() {
    use seismic_native_tests::wrong_default;
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
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
    for device in seismic_native_tests::every_device() {
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
        matches!(
            stray(Guard::Whole, anything.clone()),
            Outcome::Measured { .. }
        ),
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
        fn build(&mut self, point: usize) -> Result<PointInputs<'static>, seismic::BuildError> {
            // The device's refusal converts to a typed refusal, which is what
            // classifies the build as one that did not fit.
            let state = Tensor::zeros(self.device, Element::f32(), &[self.held[point] / 4])?;
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

    let device = Devices::discover(&[])
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
    unit.perform(&mut source, None).unwrap();
    // The second point's state is refused by the device's limit.
    let charged = device.device_ledger().charged;
    device.bound(seismic::Ledger::Own(device.device_store()), Some(charged + large / 2)).unwrap();
    let work = unit.next().unwrap();
    assert_eq!(work.stage, Stage::Discovery);
    assert_eq!(work.footprint.unpriced, 1, "an unbuilt point has no price");
    let performed = unit
        .perform(&mut source, None)
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
    device.bound(seismic::Ledger::Own(device.device_store()), None).unwrap();
    unit.perform(&mut source, None).unwrap();
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
        fn build(&mut self, point: usize) -> Result<PointInputs<'static>, seismic::BuildError> {
            let x = f32_tensor(self.device, &[self.n], &exact_values(self.n as usize));
            self.kept[point] = Some(x.clone());
            if self.clamp == Some(point) {
                self.device.bound(seismic::Ledger::Own(self.device.device_store()), Some(self.device.device_ledger().charged),
                ).unwrap();
            }
            Ok(seismic::point_inputs::<split_sum::Entry>(
                vec![split_sum::Args { x: &x }],
                seismic::WorkloadRotation::All,
                None,
                Default::default(),
            ))
        }
    }

    let device = Devices::discover(&[])
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
    unit.perform(&mut source, None).unwrap();
    source.clamp = Some(1);
    let work = unit.next().unwrap();
    assert_eq!(work.stage, Stage::Discovery);
    let performed = unit
        .perform(&mut source, None)
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
    device.bound(seismic::Ledger::Own(device.device_store()), None).unwrap();
    unit.perform(&mut source, None).unwrap();
    assert_ne!(unit.next().map(|work| work.stage), Some(Stage::Discovery));
}

/// A range on a call-time dimension forms only the launches its values can
/// activate: `gated_sum` held to `N` in 65..=1000 at `SMALL` 64 forms the
/// mirrored launch alone, sums as the portable body inside the range, refuses
/// a call outside it, and a range on a name that is not an entry dimension is
/// refused at preparation.
#[test]
fn a_ranged_kernel_forms_and_serves_only_its_range() {
    for device in seismic_native_tests::every_device() {
        let ranged = NativeSpecialization::new()
            .with_param("SMALL", 64)
            .with_range("N", 65, 1000);
        let kernel = gated_sum::native_for_device(&device, &ranged)
            .unwrap_or_else(|error| panic!("{:?}: {error}", device.backend()));
        let values = exact_values(200);
        let x = f32_tensor(&device, &[200], &values);
        let value = kernel.call(gated_sum::Args { x: &x }).unwrap().value;
        assert_eq!(
            read_f32(&value),
            [portable_gated_sum(&values)],
            "{:?}",
            device.backend()
        );
        let small = f32_tensor(&device, &[10], &exact_values(10));
        match kernel.call(gated_sum::Args { x: &small }) {
            Err(CallError::Invocation(InvocationError::OutsideRange {
                dimension,
                low: 65,
                high: 1000,
                actual: 10,
            })) => assert_eq!(dimension, "N"),
            other => panic!(
                "{:?}: N 10 outside the range: {:?}",
                device.backend(),
                other.err()
            ),
        }
        let unknown = NativeSpecialization::new()
            .with_param("SMALL", 64)
            .with_range("Q", 1, 1);
        let error = gated_sum::native_for_device(&device, &unknown)
            .err()
            .expect("Q is not a dimension of gated_sum");
        assert!(
            error.to_string().contains("`Q` is not a dimension"),
            "{error}"
        );
    }
}

/// A survey of a unit whose launches are formed separately forms them as its
/// search does, from held launch programs, and measures and validates every
/// admissible configuration: each launch's own choices, at a point where each
/// launch works. It reports how many configurations it forms before it forms
/// any, and its progress through them as it goes.
#[test]
fn a_survey_times_every_configuration_of_launches_formed_separately() {
    /// The survey's points, and the progress it reported.
    struct Reporting<'p> {
        points: Vec<TuningPoint<'p, scoped_scale::Entry>>,
        reports: Vec<seismic::SurveyProgress>,
    }
    impl<'p> seismic::PointSource<'p, scoped_scale::Entry> for Reporting<'p> {
        fn points(&self) -> Vec<seismic::PointSpec> {
            seismic::PointSource::points(&self.points)
        }
        fn build(&mut self, point: usize) -> Result<seismic::PointInputs<'p>, seismic::BuildError> {
            seismic::PointSource::build(&mut self.points, point)
        }
        fn surveyed(&mut self, progress: seismic::SurveyProgress) {
            self.reports.push(progress);
        }
    }

    let devices = Devices::discover(&[]).unwrap();
    let Ok(device) = devices.open_backend(BackendName::Metal) else {
        return;
    };
    let implementation = scoped_scale::native_implementation(&device).unwrap().unwrap();
    assert!(implementation.launch_scoped());
    let small = f32_tensor(&device, &[8], &exact_values(8));
    let large = f32_tensor(&device, &[64], &exact_values(64));
    let points = [("small", &small), ("large", &large)]
        .into_iter()
        .map(|(label, x)| TuningPoint::<scoped_scale::Entry> {
            label: label.into(),
            weight: 1.,
            class: None,
            cost: 1.,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![scoped_scale::Args { x }],
            initialize: None,
            written: Default::default(),
        })
        .collect::<Vec<_>>();
    let mut points = Reporting {
        points,
        reports: Vec::new(),
    };
    let survey = scoped_scale::native_tune(
        &device,
        &NativeSpecialization::new(),
        &mut points,
        NumericalRequirement::BitExact,
        Strategy::Survey(SurveyPlan {
            samples: 3,
            min_sample_seconds: 0.0002,
            domains: Default::default(),
        }),
    )
    .unwrap();
    // Two widths by two row counts for each of the two launches.
    assert_eq!(survey.configurations.len(), 16);
    let progress = |formed, measured| seismic::SurveyProgress {
        configurations: 16,
        formed,
        measured,
    };
    assert_eq!(points.reports.first(), Some(&progress(0, 0)));
    assert_eq!(points.reports.last(), Some(&progress(16, 16)));
    assert!(points.reports.windows(2).all(|pair| {
        pair[0].configurations == pair[1].configurations
            && pair[0].formed <= pair[1].formed
            && pair[0].measured <= pair[1].measured
            && pair[1].measured <= pair[1].formed
    }));
    for record in &survey.configurations {
        let Outcome::Measured { points, .. } = &record.outcome else {
            panic!("{:?}", record.outcome)
        };
        assert_eq!(survey.admitted(record), Ok(()));
        assert_eq!(points.len(), 2);
        assert!(points.iter().all(|point| (1..=3).contains(&point.samples.len())));
    }
    assert!(matches!(survey.method, TuningMethod::Survey { samples: 3 }));
}
