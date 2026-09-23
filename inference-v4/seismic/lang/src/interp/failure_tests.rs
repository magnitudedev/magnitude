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
        let outcome = interpreter.run(&[Arg::Tensor(shape), Arg::Tensor(dst)]).unwrap();
        assert!(matches!(
            outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(value)) if value == expected
        ));

        let mut interpreter = Interpreter::new(&entry);
        let shape = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![0.0; 2]));
        let dst = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![2.0, 3.0]));
        let outcome = interpreter.run(&[Arg::Tensor(shape), Arg::Tensor(dst)]).unwrap();
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
        .run(&[Arg::Tensor(flag), Arg::Index(2u8.into()), Arg::Range(0u8.into(), 1_000_000_000u64.into())])
        .expect("false branch does not execute the wide loop");
    assert!(matches!(
        outcome.results().next().unwrap().value(),
        OutcomeValue::Scalar(ReferenceScalar::I32(2))
    ));
    let mut interpreter = Interpreter::new(&entry);
    let flag = interpreter.add_tensor(TensorData::dense(DType::Bool, vec![1], vec![1.0]));
    let outcome = interpreter
        .run(&[Arg::Tensor(flag), Arg::Index(2u8.into()), Arg::Range(0u8.into(), 3u8.into())])
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
