//! Checked source control through ordinary preparation and native execution.
use super::*;
use seismic_compiler::feedback::FeedbackOptions;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;

fn source_products_and_failures(backend: registry::BackendName) {
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(backend).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "source-cohort.seismic".into(),
        text: r#"fn piece(choose: bool) -> tensor[1] i32:
    let mut result = tensor[1] i32
    if choose:
        let mut left = tensor[1] i32
        left[0] = 17
        result = left
    else:
        let mut right = tensor[1] i32
        right[0] = 29
        result = right
    return result

fn selected(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let chosen = piece(i < 2)
        out[i] = chosen[0]

fn doubled[N](input: &tensor[N] i32) -> tensor[N] i32:
    return input + input

fn captured_local(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[1] i32
        local[0] = 3
        let captured = local + local
        local[0] = 5
        out[i] = captured[0]

fn captured_helper(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[1] i32
        local[0] = 3
        let captured = doubled(local)
        local[0] = 5
        out[i] = captured[0]

fn captured_selected(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[1] i32
        local[0] = 3
        let mut captured = local + local
        if i < 2:
            captured = local + local
        else:
            captured = local * local
        local[0] = 5
        out[i] = captured[0]

fn captured_before_control(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[1] i32
        local[0] = 3
        let captured = local + local
        if i < 2:
            local[0] = 5
        for j in 0..2:
            local[0] = i32(j) + 7
        out[i] = captured[0]

fn captured_reduce(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[2, 1] i32
        local[0, 0] = 3
        local[1, 0] = 4
        let captured = reduce(local, 0, sum)
        local[0, 0] = 5
        out[i] = captured[0]

fn captured_view(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[2] i32
        local[0] = 3
        local[1] = 4
        let captured = reshape(local + local, (1, 2))
        local[0] = 5
        out[i] = captured[0, 0] + captured[0, 1]

fn captured_dynamic(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[i+1,i+2] i32
        local[:] = ones_like(local)
        let captured = local + local
        local[0,0] = 7
        out[i] = captured[0,0] + captured[i,i+1]

fn failing(indices: &tensor[4] i32, input: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        out[i] = 7
        let value = input[indices[i]]
        out[i] = value
"#
        .into(),
    }]))
    .unwrap();
    for (name, expected) in [
        ("selected", vec![17i32, 17, 29, 29]),
        ("captured_local", vec![6; 4]),
        ("captured_helper", vec![6; 4]),
        ("captured_selected", vec![6, 6, 9, 9]),
        ("captured_before_control", vec![6; 4]),
        ("captured_view", vec![14; 4]),
        ("captured_reduce", vec![7; 4]),
        ("captured_dynamic", vec![4; 4]),
        ("failing", vec![7, 7, 7, 7]),
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
            .unwrap_or_else(|error| panic!("{backend:?} {name} preparation: {error:?}")),
        );
        let zeros = vec![0u8; 16];
        let output = Arc::new(
            TensorInner::from_host(&device, registry::dense(DType::I32), &[4], &zeros).unwrap(),
        );
        let mut args = EncodedArgs::new();
        if name == "failing" {
            let invalid = [4i32; 4]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            args.push_tensor(Arc::new(
                TensorInner::from_host(&device, registry::dense(DType::I32), &[4], &invalid)
                    .unwrap(),
            ));
            args.push_tensor(Arc::new(
                TensorInner::from_host(&device, registry::dense(DType::I32), &[4], &zeros).unwrap(),
            ));
        }
        args.push_tensor(output.clone());
        let result = crate::api::kernel::call(&kernel, args);
        assert_eq!(result.is_err(), name == "failing", "{backend:?} {name}");
        let actual = output
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{backend:?} {name}");
    }
}

#[test]
fn cpu_source_products_and_failure_prefixes() {
    source_products_and_failures(registry::BackendName::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
fn metal_source_products_and_failure_prefixes() {
    source_products_and_failures(registry::BackendName::Metal);
}

fn source_scalar_failures(backend: registry::BackendName) {
    use crate::api::kernel::EncodedScalar;
    let catalog = crate::devices::Catalog::discover().unwrap();
    let device = catalog.open_backend(backend).unwrap();
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "source-scalar-failures.seismic".into(),
        text: r#"fn divide(x: i32, divisor: i32, out: &mut tensor[1] i32):
    out[0] = 7
    let value = x / divisor
    out[0] = value

fn remainder(x: i32, divisor: i32, out: &mut tensor[1] i32):
    out[0] = 7
    let value = x % divisor
    out[0] = value

fn divided(value: i32, divisor: i32) -> i32:
    return value / divisor

fn lanes(input: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        out[i] = 7
        let value = divided(20, input[i])
        out[i] = value

fn shifts(input: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        out[i] = 7
        let value = 1 << input[i]
        out[i] = value

fn unused_tensor(input: &tensor[4] i32, out: &mut tensor[1] i32):
    out[0] = 7
    let unused = input / input
    out[0] = 9

fn unused_local_tensor(input: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        out[i] = 7
        let unused = input / input
        out[i] = 9

fn ordered_prefix(input: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        for j in 0..4:
            out[i] = i32(j) + 7
            let value = 20 / input[j]
            out[i] = value
"#
        .into(),
    }]))
    .unwrap();
    let cases: Vec<(&str, Vec<i32>, Option<[i32; 4]>, Vec<i32>, Option<&str>)> = vec![
        ("divide", vec![20, 2], None, vec![10], None),
        (
            "divide",
            vec![20, 0],
            None,
            vec![7],
            Some("integer division by zero"),
        ),
        (
            "divide",
            vec![i32::MIN, -1],
            None,
            vec![7],
            Some("signed integer division overflow"),
        ),
        (
            "remainder",
            vec![20, 0],
            None,
            vec![7],
            Some("integer division by zero"),
        ),
        (
            "remainder",
            vec![i32::MIN, -1],
            None,
            vec![7],
            Some("signed integer division overflow"),
        ),
        (
            "lanes",
            vec![],
            Some([2, 0, 4, 0]),
            vec![10, 7, 5, 7],
            Some("integer division by zero"),
        ),
        (
            "shifts",
            vec![],
            Some([1, 32, 3, -1]),
            vec![2, 7, 8, 7],
            Some("integer shift count must be in 0..32"),
        ),
        (
            "unused_tensor",
            vec![],
            Some([2, 0, 4, 0]),
            vec![7],
            Some("integer division by zero"),
        ),
        (
            "unused_local_tensor",
            vec![],
            Some([2, 0, 4, 0]),
            vec![7; 4],
            Some("integer division by zero"),
        ),
        (
            "ordered_prefix",
            vec![],
            Some([2, 0, 4, 0]),
            vec![8; 4],
            Some("integer division by zero"),
        ),
    ];
    for (name, scalars, input, expected, error) in cases {
        eprintln!("source arithmetic case {backend:?} {name} {scalars:?}");
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
        let mut args = EncodedArgs::new();
        for value in scalars {
            args.push_scalar(EncodedScalar::I32(value));
        }
        if let Some(input) = input {
            let bytes = input
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>();
            args.push_tensor(Arc::new(
                TensorInner::from_host(&device, registry::dense(DType::I32), &[4], &bytes).unwrap(),
            ));
        }
        let output = Arc::new(
            TensorInner::from_host(
                &device,
                registry::dense(DType::I32),
                &[expected.len() as u64],
                &vec![0; expected.len() * 4],
            )
            .unwrap(),
        );
        args.push_tensor(output.clone());
        let result = crate::api::kernel::call(&kernel, args);
        if let Some(expected) = error {
            let actual = result
                .err()
                .expect("source recipe must report its failure")
                .to_string();
            assert!(actual.contains(expected), "{backend:?} {name}: {actual}");
        } else {
            result.unwrap();
        }
        let actual = output
            .read_to_host()
            .unwrap()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{backend:?} {name}");
    }
}

#[test]
fn cpu_source_arithmetic_failures_preserve_their_prefix() {
    source_scalar_failures(registry::BackendName::Cpu);
}
#[test]
#[ignore = "requires Metal device"]
fn metal_source_arithmetic_failures_preserve_their_prefix() {
    source_scalar_failures(registry::BackendName::Metal);
}
