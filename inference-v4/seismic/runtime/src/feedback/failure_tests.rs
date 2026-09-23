//! Source stops travel through native completion and the ordinary observer.
use super::*;
use seismic_compiler::feedback::{FeedbackOptions, ObservationCase, ObservationProtocol};
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::{ElementBindings, LogicalEntry};

fn validate_failure<T: TargetFamily, E: NativeExecutor<T>>(
    prepared: &Prepared<T, E>,
    device: Arc<crate::api::device::DeviceInner>,
    reference: &LogicalEntry,
    failed: bool,
) {
    let mut observer = Observer::new(prepared.device.clone(), device);
    let mut values = seismic_lang::expr::compiled::InvocationValues::new();
    let seismic_lang::entry::ParameterKind::Scalar { symbol, .. } =
        &prepared.kernel.schema().parameters()[1].kind
    else {
        panic!()
    };
    values.bind(*symbol, SymbolValue::I32(0));
    let case = ObservationCase {
        values,
        seed: 0,
        arguments: vec![
            CaseArgument::Tensor {
                representation: registry::dense(DType::I32),
                extents: vec![2],
            },
            CaseArgument::Scalar(SymbolValue::I32(0)),
        ],
    };
    let request = || ObservationRequest {
        candidate: prepared.kernel.candidate_for_variant(prepared.kernel.select(&case.values)),
        reference: reference.as_view(),
        executable: prepared.kernel.variants().first(),
        precision: &PrecisionPolicy::Exact,
        case: &case,
        protocol: ObservationProtocol {
            warmup: 0,
            trials: 1,
        },
        memory_limit: 16 * 1024 * 1024,
        reference_work_limit: 10000,
        deadline: None,
    };
    let (timing, observation) = observer
        .run_case(request(), Some(ValidationCase::Zeros))
        .unwrap();
    assert_eq!(
        timing.samples.is_empty(),
        failed,
        "only a returned invocation produces a timing sample"
    );
    let observation = observation.unwrap();
    if failed {
        assert!(matches!(
            observation.actual.termination,
            SourceTermination::Failed(_)
        ));
    } else {
        let SourceTermination::Returned(results) = &observation.actual.termination else {
            panic!("expected completed returns")
        };
        let ObservedValue::Tensor(result) = &results[0].value else {
            panic!("expected tensor result")
        };
        assert_eq!(
            result.bytes,
            [14i32, 0]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        observation.actual.inputs[0].tensor.bytes,
        [7i32, 0]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        compare_outcome(
            &observation.reference,
            &observation.actual,
            &PrecisionPolicy::Exact,
            &mut |_| Ok(())
        )
        .unwrap(),
        Comparison::Match
    );
    // A fresh private trial repeats the same prefix instead of observing a
    // restored or previously executed image.
    let again = observer.validate(request(), ValidationCase::Zeros).unwrap();
    assert_eq!(
        again.actual.inputs[0].tensor.bytes,
        observation.actual.inputs[0].tensor.bytes
    );
}
fn native_failure(backend: registry::BackendName) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(backend).unwrap();
    let module=check_source(SourceSet::new(vec![SourceFile {path:"failed-native-outcome.seismic".into(),text:
        "fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let unused = 42 / divisor\n    dst[1] = 9\n\nfn before_alloc(dst: &mut tensor[2] i32, divisor: i32):\n    parallel for i in 0..1:\n        dst[i] = 7\n        let stopped = 42 / divisor\n        let mut local = tensor[1] i32\n        local[0] = 29\n        dst[i+1] = local[0]\n\nfn before_store(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let failed_rhs = dst / divisor\n    dst[:] = failed_rhs\n\nfn returned(dst: &mut tensor[2] i32, divisor: i32) -> tensor[2] i32:\n    dst[0] = 7\n    return dst + dst\n".into()}])).unwrap();
    for (name, failed) in [("probe", true), ("before_alloc", true), ("before_store", true), ("returned", false)] {
        let id = module.entry_named(name).unwrap();
        let reference = module.entry(id, &ElementBindings::default()).unwrap();
        let kernel = crate::api::kernel::prepare(
            &module,
            id,
            ElementBindings::default(),
            &device,
            PreparationOptions::feedback(
                PrecisionPolicy::Exact,
                FeedbackOptions {
                    search_time: Duration::ZERO,
                    ..Default::default()
                },
            ),
        )
        .unwrap_or_else(|error| panic!("{backend:?} {name} preparation: {error:?}"));
        match &kernel.inner {
            crate::backends::PreparedKind::Cpu(kernel) => {
                validate_failure(&kernel.prepared, device.clone(), &reference, failed)
            }
            #[cfg(target_os = "macos")]
            crate::backends::PreparedKind::Metal(kernel) => {
                validate_failure(&kernel.prepared, device.clone(), &reference, failed)
            }
            crate::backends::PreparedKind::Cuda(_) => unreachable!(),
        }
    }
}
#[test]
fn cpu_observer_retains_failed_source_prefix() {
    native_failure(registry::BackendName::Cpu);
}
#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires Metal device"]
fn metal_observer_retains_failed_source_prefix() {
    native_failure(registry::BackendName::Metal);
}

fn reached_private_allocation_preserves_prefix(source: &str) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "reached-private.seismic".into(),
        text: source.into(),
    }])).unwrap();
    let kernel = crate::api::kernel::prepare(
        &module, module.entry_named("main").unwrap(), ElementBindings::default(), &device,
        PreparationOptions::feedback(PrecisionPolicy::Exact, FeedbackOptions {
            search_time: Duration::ZERO, ..Default::default()
        }),
    ).unwrap();
    let output = Arc::new(TensorInner::from_host(
        &device, registry::dense(DType::I32), &[1], &0i32.to_le_bytes(),
    ).unwrap());
    let baseline = device.memory_usage().charged;
    device.set_memory_limit(Some(baseline + 4096)).unwrap();
    let mut args = EncodedArgs::new();
    args.push_tensor(output.clone());
    args.push_scalar(crate::api::kernel::EncodedScalar::I32(0));
    let error = crate::api::kernel::call(&Arc::new(kernel), args).err().expect("division fails after the first allocation");
    assert!(matches!(error, crate::api::CallError::Execution(ExecutionError::DataCheckFailed(_))), "{error:?}");
    assert_eq!(output.read_to_host().unwrap(), 7i32.to_le_bytes());
    assert_eq!(device.memory_usage().charged, baseline, "failed run released its private backing");
}

#[test]
fn cpu_reached_private_allocation_does_not_reserve_unvisited_loop_envelope() {
    reached_private_allocation_preserves_prefix("fn main(out: &mut tensor[1] i32, divisor: i32):\n    for i in 0..65536:\n        let mut scratch = tensor[i+1] i32\n        scratch[0] = 7\n        out[0] = scratch[0]\n        let quotient = 1 / divisor\n        out[0] = quotient\n");
}

#[test]
fn cpu_reached_private_allocation_skips_backing_after_source_failure() {
    reached_private_allocation_preserves_prefix("fn main(out: &mut tensor[1] i32, divisor: i32):\n    let mut first = tensor[1] i32\n    first[0] = 7\n    out[0] = first[0]\n    let quotient = 1 / divisor\n    let mut later = tensor[65536] i32\n    later[0] = quotient\n    out[0] = later[0]\n");
}

// These exercise the required schedule placement. Native-segment local
// reservation envelopes are a different physical placement's contract.
fn reached_geometry_outcome(source: &str, with_inputs: bool, expected: &[i32]) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "reached-geometry.seismic".into(), text: source.into(),
    }])).unwrap();
    let kernel = Arc::new(crate::api::kernel::prepare(
        &module, module.entry_named("probe").unwrap(), ElementBindings::default(), &device,
        PreparationOptions::feedback(PrecisionPolicy::Exact, FeedbackOptions {
            search_time: Duration::ZERO, ..Default::default()
        }),
    ).unwrap());
    let tensor = |shape: &[u64], data: &[i32]| Arc::new(TensorInner::from_host(
        &device, registry::dense(DType::I32), shape,
        &data.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
    ).unwrap());
    let output = tensor(&[4], &[0; 4]);
    let input = tensor(&[8, 2], &[0; 16]);
    let visible = tensor(&[4, 2], &[0, 0, 1, 4, 0, 8, 4, 5]);
    let baseline = device.memory_usage().charged;
    device.set_memory_limit(Some(baseline + 4096)).unwrap();
    let mut args = EncodedArgs::new();
    if with_inputs {
        args.push_tensor(input.clone());
        args.push_tensor(visible.clone());
    }
    args.push_tensor(output.clone());
    crate::api::kernel::call(&kernel, args).unwrap();
    let bytes = output.read_to_host().unwrap();
    let actual = bytes.chunks_exact(4).map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(device.memory_usage().charged, baseline, "completed region releases private instances");
}

#[test]
fn cpu_reached_helper_geometry_follows_actual_slice() {
    reached_geometry_outcome(r#"fn seed[M,K](x: &tensor[M,K] i32) -> i32:
    let mut scratch = tensor[M,K] i32
    scratch[:] = ones_like(scratch)
    return reduce(reduce(scratch, 1, sum), 0, sum)

fn probe(input: &tensor[8,2] i32, visible: &tensor[4,2] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let lo = visible[i,0]
        let hi = visible[i,1]
        if hi > lo:
            out[i] = seed(input[lo:hi,:])
"#, true, &[0, 6, 16, 2]);
}

#[test]
fn cpu_reached_snapshot_preserves_varying_axes_and_old_contents() {
    reached_geometry_outcome(r#"fn probe(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[i+1,i+2] i32
        local[:] = ones_like(local)
        let captured = local + local
        local[0,0] = 7
        out[i] = captured[0,0] + captured[i,i+1]
"#, false, &[4, 4, 4, 4]);
}

#[test]
fn cpu_reached_helper_geometry_composes_transpose_empty_and_full_views() {
    reached_geometry_outcome(r#"fn seed[N](x: &tensor[N] i32) -> i32 where N >= 0:
    let scratch = ones_like(x)
    return reduce(scratch, 0, sum)

fn probe(input: &tensor[8,2] i32, visible: &tensor[4,2] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let lo = visible[i,0]
        let hi = visible[i,1]
        let moved = input.T
        let dynamic = seed(moved[1,lo:hi])
        let empty = seed(input[0:0,1])
        let full = seed(input[:,1])
        out[i] = dynamic + empty + full
"#, true, &[8, 11, 16, 9]);
}

#[test]
fn cpu_reached_branch_result_keeps_selected_instance_across_visits() {
    reached_geometry_outcome(r#"fn probe(out: &mut tensor[4] i32):
    for i in 0..4:
        let mut selected = tensor[2] i32
        selected[:] = ones_like(selected)
        if i % 2 == 0:
            let mut replacement = tensor[2] i32
            replacement[:] = ones_like(replacement) * 7
            selected = replacement
        out[i] = selected[0]
"#, false, &[7, 1, 7, 1]);
}
