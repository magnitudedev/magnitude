use super::*;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;

fn scalar_reductions(backend: registry::BackendName) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(backend).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "scalar-reduction.seismic".into(),
        text: "fn first[N](x: &tensor[N] f32) -> f32:\n    return x[0]\n\nfn dependent[N](x: &tensor[N] f32) -> f32 where N >= 2:\n    return x[0] + x[1] + x[0] * 2.0\n\nfn total[N](x: &tensor[N] f32) -> f32 where N >= 0:\n    return reduce(x, 0, sum)\n\nfn expression[N](x: &tensor[N] f32) -> f32 where N >= 0:\n    return reduce(x + x, 0, sum)\n\nfn snapshot[N](x: &mut tensor[N] f32) -> f32:\n    let a = x + x\n    parallel for i in 0..N:\n        x[i] = 0.0\n    return reduce(a, 0, sum)\n\nfn narrow[N](x: &tensor[N] f32) -> f32:\n    return reduce(f32(f16(x)), 0, sum)\n\nfn transformed(x: &tensor[4] f32) -> f32:\n    let a = reshape(x + x, (2, 2))\n    return reduce(reduce(a, 1, sum), 0, sum)\n\nfn nested(x: &tensor[2,3] f32) -> f32:\n    return reduce(reduce(x, 1, sum), 0, sum)\n".into(),
    }])).unwrap();
    for (name, extents, data, expected) in [
        ("first", vec![2], vec![3.0f32, 7.0], 3.0f32),
        ("dependent", vec![2], vec![3.0f32, 7.0], 16.0f32),
        ("total", vec![4], vec![1.0f32, -2.0, 3.0, 4.0], 6.0f32),
        ("total", vec![0], vec![], 0.0),
        ("expression", vec![4], vec![1.0, -2.0, 3.0, 4.0], 12.0),
        ("expression", vec![0], vec![], 0.0),
        ("snapshot", vec![4], vec![1.0, -2.0, 3.0, 4.0], 12.0),
        ("narrow", vec![3], vec![2049.0, 1.0, -2048.0], 1.0),
        ("transformed", vec![4], vec![1.0, -2.0, 3.0, 4.0], 12.0),
        (
            "nested",
            vec![2, 3],
            vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0],
            5.0,
        ),
    ] {
        let kernel = Arc::new(
            crate::api::kernel::prepare(
                &module,
                module.entry_named(name).unwrap(),
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
            .unwrap(),
        );
        let bytes = data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = Arc::new(
            TensorInner::from_host(&device, registry::dense(DType::F32), &extents, &bytes).unwrap(),
        );
        let mut args = EncodedArgs::new();
        args.push_tensor(tensor);
        let mut output = crate::api::kernel::call(&kernel, args)
            .unwrap_or_else(|error| panic!("{backend:?} {name} {extents:?}: {error:?}"));
        let ArgumentValue::F32(actual) = output.take_scalar() else {
            panic!("scalar reduction returned another category")
        };
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{backend:?} {name} {extents:?}"
        );
    }
}

#[test]
fn cpu_scalar_reductions_publish_scalar_slots_and_empty_sum_identity() {
    scalar_reductions(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_scalar_reductions_publish_scalar_slots_and_empty_sum_identity() {
    scalar_reductions(registry::BackendName::Metal);
}

fn range_endpoints(backend: registry::BackendName) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(backend).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "range-endpoints.seismic".into(),
        text: "fn probe(x: &tensor[4] f32, times: range[4]) -> f32:\n    return sum_range(x, times)\n\nfn element(x: &tensor[4] f32, i: index[4]) -> f32:\n    return x[i]\n\nfn sum_range(x: &tensor[4] f32, times: range[4]) -> f32:\n    let mut total = 0.0\n    for i in times:\n        total = total + element(x, i)\n    return total\n".into(),
    }])).unwrap();
    let kernel = Arc::new(
        crate::api::kernel::prepare(
            &module,
            module.entry_named("probe").unwrap(),
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
        .unwrap(),
    );
    let bytes = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let tensor = Arc::new(
        TensorInner::from_host(&device, registry::dense(DType::F32), &[4], &bytes).unwrap(),
    );
    for (start, end, expected) in [
        (0, 0, 0.0f32),
        (2, 2, 0.0),
        (0, 1, 1.0),
        (1, 3, 5.0),
        (0, 4, 10.0),
    ] {
        let mut args = EncodedArgs::new();
        args.push_tensor(tensor.clone());
        args.push_scalar(crate::api::kernel::EncodedScalar::Range {
            start: u64::try_from(start).unwrap().into(),
            end: u64::try_from(end).unwrap().into(),
        });
        let mut result = crate::api::kernel::call(&kernel, args).unwrap();
        let ArgumentValue::F32(actual) = result.take_scalar() else {
            panic!("range loop returned non-scalar")
        };
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{backend:?} {start}..{end}"
        );
    }
}

#[test]
fn cpu_range_loop_uses_invocation_endpoints() {
    range_endpoints(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_range_loop_uses_invocation_endpoints() {
    range_endpoints(registry::BackendName::Metal);
}
