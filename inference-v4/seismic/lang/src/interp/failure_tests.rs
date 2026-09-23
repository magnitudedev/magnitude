use super::*;
use crate::checked::{check_source, SourceFile, SourceSet};
use crate::entry::ElementBindings;
use crate::reference_math::ScalarFailure;

fn entry(source: &str) -> LogicalEntry {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "failed-outcome.seismic".into(),
        text: source.into(),
    }]))
    .unwrap();
    module
        .entry(
            module.entry_named("probe").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap()
}
fn run(entry: &LogicalEntry, scalar: i32) -> OracleOutcome {
    let mut interpreter = Interpreter::new(entry);
    let dst = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![2., 3.]));
    interpreter
        .run(&[Arg::Tensor(dst), Arg::Scalar(ReferenceScalar::I32(scalar))])
        .unwrap()
}

#[test]
fn quantity_product_remains_exact_above_u64_until_explicit_word_cast() {
    let entry = entry("fn probe(i: index[9223372036854775807]) -> i32:\n    let product = i * i\n    return i32(product / (i + 1))\n");
    let outcome = Interpreter::new(&entry)
        .run(&[Arg::Index(4_294_967_296u64.into())])
        .expect("exact mathematical product is a legal source value");
    let result = outcome.results().next().expect("scalar result");
    assert!(matches!(
        result.value(),
        OutcomeValue::Scalar(ReferenceScalar::I32(-1))
    ));
}

#[test]
fn signed_quantity_division_and_remainder_fail_at_their_source_position() {
    for (operation, expected) in [("/", 2), ("%", 0)] {
        let entry = entry(&format!("fn probe[N](shape: &tensor[N] f32, dst: &mut tensor[2] i32) -> i32:\n    dst[0] = 7\n    let q = (N - 3) {operation} (N - 2)\n    dst[1] = 9\n    return i32(q)\n"));
        let mut interpreter = Interpreter::new(&entry);
        let shape = interpreter.add_tensor(TensorData::dense(DType::F32, vec![1], vec![0.0]));
        let dst = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![2.0, 3.0]));
        let outcome = interpreter
            .run(&[Arg::Tensor(shape), Arg::Tensor(dst)])
            .unwrap();
        assert!(matches!(
            outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(value)) if value == expected
        ));

        let mut interpreter = Interpreter::new(&entry);
        let shape = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![0.0; 2]));
        let dst = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![2.0, 3.0]));
        let outcome = interpreter
            .run(&[Arg::Tensor(shape), Arg::Tensor(dst)])
            .unwrap();
        assert_eq!(
            failure(&outcome).cause,
            SourceFailureCause::Scalar(ScalarFailure::IntegerDivisionByZero)
        );
        let dst_input = outcome.inputs().nth(1).unwrap();
        let dst = dst_input.tensor();
        assert_eq!(dst.read(0).unwrap(), 7.0);
        assert_eq!(dst.read(1).unwrap(), 3.0);
    }
}

#[test]
fn guarded_quantity_loop_uses_joined_actual_value_without_visiting_false_arm() {
    let entry = entry("fn probe(flag: &tensor[1] bool, start: index[1000000000], times: range[1000000000]) -> i32:\n    let mut q = start + 0\n    if flag[0]:\n        for i in times:\n            q = q * (i + 1)\n    return i32(q)\n");
    let mut interpreter = Interpreter::new(&entry);
    let flag = interpreter.add_tensor(TensorData::dense(DType::Bool, vec![1], vec![0.0]));
    let outcome = interpreter
        .run(&[
            Arg::Tensor(flag),
            Arg::Index(2u8.into()),
            Arg::Range(0u8.into(), 1_000_000_000u64.into()),
        ])
        .expect("false branch does not execute the wide loop");
    assert!(matches!(
        outcome.results().next().unwrap().value(),
        OutcomeValue::Scalar(ReferenceScalar::I32(2))
    ));
    let mut interpreter = Interpreter::new(&entry);
    let flag = interpreter.add_tensor(TensorData::dense(DType::Bool, vec![1], vec![1.0]));
    let outcome = interpreter
        .run(&[
            Arg::Tensor(flag),
            Arg::Index(2u8.into()),
            Arg::Range(0u8.into(), 3u8.into()),
        ])
        .expect("loop carry advances the actual quantity each visit");
    assert!(matches!(
        outcome.results().next().unwrap().value(),
        OutcomeValue::Scalar(ReferenceScalar::I32(12))
    ));
}
fn failure(outcome: &OracleOutcome) -> &SourceFailure {
    let SourceTermination::Failed(failure) = outcome.termination() else {
        panic!("expected source stop")
    };
    assert_eq!(outcome.results().len(), 0);
    failure
}
#[test]
fn failure_owns_only_completed_writes_and_actual_helper_site() {
    let entry = entry("fn quotient(divisor: i32) -> i32:\n    return 42 / divisor\n\nfn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let unused = quotient(divisor)\n    dst[1] = 9\n");
    let outcome = run(&entry, 0);
    let stopped = failure(&outcome);
    assert_eq!(
        stopped.cause,
        SourceFailureCause::Scalar(ScalarFailure::IntegerDivisionByZero)
    );
    let (_, function) = entry
        .program()
        .functions()
        .find(|(_, function)| function.stable() == stopped.event.body())
        .unwrap();
    assert!(function.name().contains("quotient"));
    assert!(function.nodes(function.root()).any(|(id, node)| node
        .events()
        .iter()
        .any(|e| matches!(e.kind(), crate::entry::SemanticEventKind::MayFail))
        && SourceFailure::at(function, id, stopped.cause.clone()) == *stopped));
    let input = outcome.inputs().next().unwrap();
    assert_eq!(input.tensor().read(0).unwrap(), 7.);
    assert_eq!(input.tensor().read(1).unwrap(), 3.);
    assert!(outcome.relation().is_none());
}
#[test]
fn recipe_causes_and_ordered_prefix_are_preserved() {
    for (expression, arg, cause) in [
        ("42 / divisor", 0, ScalarFailure::IntegerDivisionByZero),
        (
            "(-2147483647 - 1) / divisor",
            -1,
            ScalarFailure::SignedDivisionOverflow,
        ),
        ("42 << divisor", 32, ScalarFailure::ShiftCount),
    ] {
        let entry = entry(&format!("fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    for i in 0..2:\n        dst[i] = 7\n        let unused = {expression}\n    dst[1] = 9\n"));
        let outcome = run(&entry, arg);
        assert_eq!(failure(&outcome).cause, SourceFailureCause::Scalar(cause));
        let input = outcome.inputs().next().unwrap();
        assert_eq!(input.tensor().read(0).unwrap(), 7.);
        assert_eq!(input.tensor().read(1).unwrap(), 3.);
    }
}
#[test]
fn unused_partial_tensor_stops_at_its_definition() {
    let entry = entry("fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let unused = dst / divisor\n    dst[1] = 9\n");
    let outcome = run(&entry, 0);
    assert_eq!(
        failure(&outcome).cause,
        SourceFailureCause::Scalar(ScalarFailure::IntegerDivisionByZero)
    );
    assert_eq!(
        outcome.inputs().next().unwrap().tensor().read(1).unwrap(),
        3.
    );
}
#[test]
fn entered_parallel_regions_describe_possible_prefixes_only_when_executed() {
    let entry = entry("fn probe(dst: &mut tensor[2] i32, enabled: i32):\n    if enabled > 0:\n        parallel for i in 0..2:\n            dst[i] = 7\n            let unused = 42 / (enabled - 1)\n");
    assert!(run(&entry, 0).relation().is_none());
    let stopped = run(&entry, 1);
    failure(&stopped);
    assert_eq!(stopped.relation().unwrap().parallel_regions().len(), 1);
    assert!(stopped.relation().unwrap().associations().is_empty());
}
#[test]
fn invocation_and_service_failures_are_not_source_stops() {
    let entry=entry("fn probe(dst: &mut tensor[2] i32, divisor: i32):\n    dst[0] = 7\n    let unused = 42 / divisor\n");
    assert!(matches!(
        Interpreter::new(&entry).run(&[]),
        Err(OracleError::InvalidInvocation(_))
    ));
    for memory in [false, true] {
        let mut interpreter = Interpreter::new(&entry);
        let dst = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![0.; 2]));
        let result = interpreter.run_bounded_with_memory(
            &[Arg::Tensor(dst), Arg::Scalar(ReferenceScalar::I32(0))],
            if memory { u64::MAX } else { 0 },
            if memory { 0 } else { u64::MAX },
        );
        assert!(if memory {
            matches!(result, Err(OracleError::MemoryLimit { .. }))
        } else {
            matches!(result, Err(OracleError::WorkLimit { .. }))
        });
    }
}

fn reduced(source: &str, input: TensorData) -> ReferenceScalar {
    let entry = entry(source);
    let mut interpreter = Interpreter::new(&entry);
    let input = interpreter.add_tensor(input);
    let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
    let OutcomeValue::Scalar(value) = outcome.results().next().unwrap().value() else {
        panic!("scalar reduction result")
    };
    value
}

#[test]
fn argmax_compares_in_the_input_dtype() {
    let argmax = "fn probe[N](x: &tensor[N] f32) -> i32:\n    return reduce(x, 0, argmax)\n";
    for (values, expected) in [
        (vec![0.3, 0.7], 1),
        (vec![0.7, 0.3], 0),
        (vec![0.5, 0.5], 0),
        (vec![-2.5, -2.25, -2.75], 1),
    ] {
        let input = TensorData::dense(DType::F32, vec![values.len()], values);
        assert_eq!(reduced(argmax, input), ReferenceScalar::I32(expected));
    }
    let argmax = "fn probe[N](x: &tensor[N] f16) -> i32:\n    return reduce(x, 0, argmax)\n";
    let input = TensorData::dense(DType::F16, vec![3], vec![0.25, 0.75, 0.5]);
    assert_eq!(reduced(argmax, input), ReferenceScalar::I32(1));
}

#[test]
fn extremum_reductions_fold_from_the_first_element() {
    let nan = f32::from_bits(0x7fc0_0000) as f64;
    let max = "fn probe[N](x: &tensor[N] f32) -> f32:\n    return reduce(x, 0, max)\n";
    let input = TensorData::dense(DType::F32, vec![2], vec![nan, nan]);
    assert_eq!(reduced(max, input), ReferenceScalar::F32(0x7fc0_0000));
    let argmax = "fn probe[N](x: &tensor[N] f32) -> i32:\n    return reduce(x, 0, argmax)\n";
    let input = TensorData::dense(DType::F32, vec![2], vec![nan, 1.0]);
    assert_eq!(reduced(argmax, input), ReferenceScalar::I32(0));
    let min = "fn probe[N](x: &tensor[N] i32) -> i32:\n    return reduce(x, 0, min)\n";
    let input = TensorData::dense(DType::I32, vec![3], vec![4.0, -7.0, 2.0]);
    assert_eq!(reduced(min, input), ReferenceScalar::I32(-7));
}

#[test]
fn callee_dimension_extents_bind_in_the_callee_frame() {
    let single = entry("fn twice[N](x: &tensor[N] i32) -> tensor[N] i32:\n    return x + x\n\nfn probe(input: &tensor[4] i32) -> tensor[4] i32:\n    return twice(input * 1)\n");
    let mut interpreter = Interpreter::new(&single);
    let input =
        interpreter.add_tensor(TensorData::dense(DType::I32, vec![4], vec![0., 1., 2., 3.]));
    let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(result) = result.value() else {
        panic!("tensor result")
    };
    assert_eq!(
        (0..4).map(|i| result.read(i).unwrap()).collect::<Vec<_>>(),
        [0., 2., 4., 6.]
    );

    // One family, two call sites with different actual dimensions: each call
    // frame binds its own formal.
    let two_sites = entry("fn twice[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return x + x\n\nfn probe[N, M](a: &tensor[N] f32, b: &tensor[M] f32) -> (tensor[N] f32, tensor[M] f32):\n    return (twice(a * 1.0), twice(b * 1.0))\n");
    let mut interpreter = Interpreter::new(&two_sites);
    let a = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![1., 2.]));
    let b = interpreter.add_tensor(TensorData::dense(DType::F32, vec![3], vec![3., 4., 5.]));
    let outcome = interpreter.run(&[Arg::Tensor(a), Arg::Tensor(b)]).unwrap();
    let results = outcome
        .results()
        .map(|result| {
            let OutcomeValue::Tensor(tensor) = result.value() else {
                panic!("tensor result")
            };
            (0..tensor.element_count())
                .map(|i| tensor.read(i).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(results, [vec![2., 4.], vec![6., 8., 10.]]);
}
