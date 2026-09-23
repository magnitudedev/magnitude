//! Exact monomorphization of checked templates into one entry-owned semantic
//! program and expression arena.

#[cfg(test)]
mod source_check_tests {
    use crate::checked::{check_source, SourceFile, SourceSet};
    use crate::entry::{ElementBindings, SemanticNodeView};
    use crate::expr::{AnyExpr, NodeView};

    #[test]
    fn slice_axes_retain_checked_omission_premises_in_their_source_scope() {
        use crate::entry::{SliceAxis, ViewTransform};
        for (source, expects_omission) in [
            ("fn run[N](dst: &mut tensor[N] f32):\n    for i in 0..N:\n        dst[i:i+1] = zeros_like(dst[i:i+1])\n", true),
            ("fn run(dst: &tensor[4] f32, lo: i32, hi: i32):\n    let view = dst[lo:hi]\n", false),
        ] {
            let module = check_source(SourceSet::new(vec![SourceFile { path: "slice-premise.seismic".into(), text: source.into() }])).unwrap();
            let entry = module.entry(module.entry_named("run").unwrap(), &ElementBindings::default()).unwrap();
            let program = entry.program();
            let function = program.function(program.family(program.root()).reference().function());
            let mut regions = vec![function.root()];
            let mut omitted = false;
            let mut checked = false;
            while let Some(region) = regions.pop() {
                for (_, node) in function.nodes(region) {
                    match node.view() {
                        SemanticNodeView::Loop { body, .. } => regions.push(body),
                        SemanticNodeView::If { then, otherwise, .. } => regions.extend([then, otherwise]),
                        SemanticNodeView::View { transform: ViewTransform::Slice { axes }, .. } => {
                            for axis in axes {
                                if let SliceAxis::Range { check_start, check_order, check_end, .. } = axis {
                                    omitted |= !check_start && !check_order && !check_end;
                                    checked |= *check_start && *check_order && *check_end;
                                }
                            }
                        }
                        _ => {},
                    }
                }
            }
            assert_eq!(omitted, expects_omission, "{source}");
            assert!(checked, "runtime slice checks must remain attached to their actual operands");
        }
    }

    #[test]
    fn invocation_known_operation_checks_follow_prior_writes() {
        for expression in ["42 / divisor", "dst[divisor]", "42 / (divisor + 1)"] {
            let module = check_source(SourceSet::new(vec![SourceFile {
                path: "source-check.seismic".into(),
                text: format!(
                    "fn run(dst: &mut tensor[1] i32, divisor: i32) -> i32:\n    dst[0] = 7\n    return {expression}\n"
                ),
            }]))
            .unwrap();
            let entry = module
                .entry(
                    module.entry_named("run").unwrap(),
                    &ElementBindings::default(),
                )
                .unwrap();
            assert!(
                matches!(
                    entry
                        .arena()
                        .view(AnyExpr::Bool(entry.domain().predicate().node())),
                    NodeView::BoolConst(true)
                ),
                "{expression}: an implicit operation check narrowed entry admission"
            );
            let program = entry.program();
            let function = program.function(program.family(program.root()).reference().function());
            let nodes = function.nodes(function.root()).collect::<Vec<_>>();
            let write = nodes
                .iter()
                .position(|(_, node)| matches!(node.view(), SemanticNodeView::ElementWrite { .. }))
                .expect("source write is present");
            let failures = nodes
                .iter()
                .enumerate()
                .filter_map(|(ordinal, (_, node))| {
                    matches!(
                        node.view(),
                        SemanticNodeView::Check { .. }
                            | SemanticNodeView::Primitive {
                                primitive: crate::intrinsics::PrimitiveId::Binary(
                                    crate::syntax::ast::BinaryOp::Div
                                ),
                                ..
                            }
                    )
                    .then_some(ordinal)
                })
                .collect::<Vec<_>>();
            assert!(
                !failures.is_empty(),
                "{expression}: source failure operation disappeared"
            );
            assert!(
                failures.iter().all(|failure| *failure > write),
                "{expression}: source failure moved before the write"
            );
            let mut interpreter = crate::interp::Interpreter::new(&entry);
            let dst = interpreter.add_tensor(crate::interp::TensorData::dense(
                crate::types::DType::I32,
                vec![1],
                vec![0.0],
            ));
            let divisor = if expression == "42 / (divisor + 1)" {
                -1
            } else if expression == "dst[divisor]" {
                1
            } else {
                0
            };
            let outcome = interpreter.run(&[
                crate::interp::Arg::Tensor(dst),
                crate::interp::Arg::Scalar(crate::reference_math::ReferenceScalar::I32(divisor)),
            ]);
            let outcome = outcome.expect("source stop is a completed invocation");
            let crate::failure::SourceTermination::Failed(failure) = outcome.termination() else {
                panic!("source operation must fail at execution");
            };
            let expected = if expression == "dst[divisor]" {
                crate::failure::SourceFailureCause::Check(crate::entry::CheckReason::IndexBound)
            } else {
                crate::failure::SourceFailureCause::Scalar(
                    crate::reference_math::ScalarFailure::IntegerDivisionByZero,
                )
            };
            assert_eq!(failure.cause, expected);
            assert_eq!(outcome.results().len(), 0);
            assert_eq!(
                outcome.inputs().next().unwrap().tensor().read(0).unwrap(),
                7.0
            );
            assert!(function.nodes(function.root()).any(|(id, node)| node
                .events()
                .iter()
                .any(|e| matches!(e.kind(), crate::entry::SemanticEventKind::MayFail))
                && crate::failure::SourceFailure::at(function, id, failure.cause.clone())
                    == *failure));
        }
    }

    #[test]
    fn operation_failures_participate_in_the_actual_event_predecessor_chain() {
        use crate::entry::SemanticEventKind;
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "failure-events.seismic".into(),
            text: r#"fn quotient(divisor: i32) -> i32:
    return 42 / divisor

fn scalar(dst: &mut tensor[1] i32, divisor: i32):
    dst[0] = 7
    let unused = 42 / divisor
    dst[0] = 9

fn helper(dst: &mut tensor[1] i32, divisor: i32):
    dst[0] = 7
    let unused = quotient(divisor)
    dst[0] = 9

fn tensor_case(dst: &mut tensor[1] i32, src: &tensor[2] i32):
    dst[0] = 7
    let unused = src / src
    dst[0] = 9

fn bound(dst: &mut tensor[1] i32, index: i32):
    dst[0] = 7
    let unused = dst[index]
    dst[0] = 9
"#
            .into(),
        }]))
        .unwrap();
        for name in ["scalar", "helper", "tensor_case", "bound"] {
            let entry = module
                .entry(
                    module.entry_named(name).unwrap(),
                    &ElementBindings::default(),
                )
                .unwrap();
            let program = entry.program();
            let function = program.function(program.family(program.root()).reference().function());
            let events = function
                .nodes(function.root())
                .flat_map(|(_, node)| node.events())
                .collect::<Vec<_>>();
            let failure = events
                .iter()
                .position(|e| matches!(e.kind(), SemanticEventKind::MayFail))
                .expect("source operation owns its failure event");
            let first_write = events
                .iter()
                .position(|e| matches!(e.kind(), SemanticEventKind::Write(_)))
                .unwrap();
            let last_write = events
                .iter()
                .rposition(|e| matches!(e.kind(), SemanticEventKind::Write(_)))
                .unwrap();
            assert!(
                first_write < failure && failure < last_write,
                "{name}: failure must stay between the writes"
            );
            assert!(events[failure].place().is_none());
            assert!(events[failure].representation().is_none());
            for adjacent in events.windows(2) {
                assert!(
                    adjacent[1]
                        .ordering_dependencies()
                        .contains(&adjacent[0].id()),
                    "{name}: source effect lost its predecessor"
                );
            }
        }
    }

    #[test]
    fn explicit_parameter_bounds_remain_entry_preconditions() {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "bounded-parameter.seismic".into(),
            text: "fn run(i: index[4]) -> index[4]:\n    return i\n".into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("run").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        assert!(!matches!(
            entry
                .arena()
                .view(AnyExpr::Bool(entry.domain().predicate().node())),
            NodeView::BoolConst(true)
        ));
    }
}

#[cfg(test)]
mod result_schema_tests {
    use super::*;
    use crate::checked::{check_source, SourceFile, SourceSet};
    use crate::interp::{Arg, Interpreter, OutcomeValue, TensorData};
    use crate::reference_math::ReferenceScalar;

    fn module(source: &str) -> crate::checked::CheckedModule {
        check_source(SourceSet::new(vec![SourceFile {
            path: "result-schema.seismic".into(),
            text: source.into(),
        }])).unwrap()
    }

    #[test]
    fn declared_result_paths_survive_semantic_leaf_flattening() {
        let module = module("fn probe(x: &tensor[1] f32, pair: (f32, range[4])) -> (tensor[1] f32, (f32, range[4])):\n    return (load(x), pair)\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let paths = entry.schema().results().iter().map(|result| result.path.clone()).collect::<Vec<_>>();
        assert_eq!(paths, vec![vec![0], vec![1, 0], vec![1, 1]]);
    }

    #[test]
    fn declared_result_geometry_does_not_capture_body_runtime_values() {
        let module = module("fn probe(x: &tensor[2,3] f32) -> tensor[2,3] f32:\n    let mut y = load(x)\n    for i in 0..2:\n        y = reshape(y.T, (2,3))\n    return y\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let ResultKind::Tensor { axes, .. } = &entry.schema().results()[0].kind else { panic!("tensor result") };
        for axis in axes {
            assert!(entry.arena().free_symbols(AnyExpr::Nat(*axis)).is_empty());
        }
    }

    #[test]
    fn return_rounding_is_an_ordered_semantic_producer() {
        let module = module("fn scalar(x: f32) -> f16:\n    return x\n\nfn tensor(x: &tensor[2] f32) -> tensor[2] f16:\n    return load(x)\n");
        for (name, expected) in [("scalar", SemanticType::Scalar(DType::F16)), ("tensor", SemanticType::Tensor(TensorSemantics { representation: crate::registry::dense(DType::F16), axes: vec![], storage: TensorStorage::Computed }))] {
            let entry = module.entry(module.entry_named(name).unwrap(), &ElementBindings::default()).unwrap();
            let candidate = entry.program().family(entry.program().root()).candidates().iter().find(|candidate| candidate.numerical == NumericalRole::Reference).unwrap();
            let function = entry.program().function(candidate.function);
            let result = function.results()[0];
            match (&function.value(result).ty, &expected) {
                (SemanticType::Scalar(actual), SemanticType::Scalar(expected)) => assert_eq!(actual, expected),
                (SemanticType::Tensor(actual), SemanticType::Tensor(expected)) => assert_eq!(actual.representation, expected.representation),
                _ => panic!("return conversion changed value category"),
            }
            assert!(matches!(function.value(result).origin, ValueOrigin::Node(_)));
        }
    }

    #[test]
    fn declared_shape_error_and_concrete_conversion_error_keep_return_position() {
        let shape_source = "fn probe(x: &tensor[2] f32) -> tensor[3] f32:\n    return load(x)\n";
        let errors = check_source(SourceSet::new(vec![SourceFile {
            path: "bad-result-shape.seismic".into(),
            text: shape_source.into(),
        }])).unwrap_err();
        let error = &errors.diagnostics().items()[0];
        assert_eq!(error.location.path, "bad-result-shape.seismic");
        assert!(error.location.span.start >= shape_source.find("load(x)").unwrap() as u32);

        let conversion_source = "fn probe(x: &tensor[1] U) -> tensor[1] T:\n    return load(x)\n";
        let module = module(conversion_source);
        let bindings = ElementBindings::new()
            .bind("U", crate::registry::dense(DType::I32))
            .bind("T", crate::registry::dense(DType::F32));
        let errors = module.entry(module.entry_named("probe").unwrap(), &bindings).unwrap_err();
        let error = &errors.diagnostics().items()[0];
        assert!(error.message.contains("no defined source installation conversion"));
        assert!(error.location.span.start >= conversion_source.find("load(x)").unwrap() as u32);
    }

    #[test]
    fn scalar_installations_and_compound_assignment_preserve_rounding_order() {
        let module = module("fn assigned(x: f32) -> f16:\n    let mut y = f16(0.0)\n    y = x\n    return y\n\nfn called(x: f32) -> f16:\n    return helper(x)\n\nfn helper(x: f16) -> f16:\n    return x\n\nfn added(x: f32) -> f16:\n    let mut y = f16(1.0)\n    y += x\n    return y\n\nfn element_added(x: f32) -> tensor[1] f16:\n    let mut y = tensor[1] f16\n    y[0] = f16(1.0)\n    y[0] += x\n    return y\n");
        let input = ReferenceScalar::F32(0x3a00_0800);
        for name in ["assigned", "called", "added"] {
            let entry = module.entry(module.entry_named(name).unwrap(), &ElementBindings::default()).unwrap();
            let outcome = Interpreter::new(&entry).run(&[Arg::Scalar(input)]).unwrap();
            let OutcomeValue::Scalar(ReferenceScalar::F16(bits)) = outcome.results().next().unwrap().value() else {
                panic!("{name} did not return f16")
            };
            assert_eq!(bits, if name == "added" { 0x3c01 } else { 0x1000 }, "{name}");
        }
        let entry = module.entry(module.entry_named("element_added").unwrap(), &ElementBindings::default()).unwrap();
        let outcome = Interpreter::new(&entry).run(&[Arg::Scalar(input)]).unwrap();
        let result = outcome.results().next().unwrap();
        let OutcomeValue::Tensor(tensor) = result.value() else { panic!("tensor result") };
        assert_eq!(tensor.canonical_bytes().unwrap().unwrap(), &0x3c01u16.to_le_bytes());
    }

    #[test]
    fn tensor_assignment_and_store_install_the_destination_element_type() {
        let module = module("fn assigned(x: &tensor[2] f32) -> tensor[2] f16:\n    let mut y = tensor[2] f16\n    y = load(x)\n    return y\n\nfn stored(x: &tensor[2] f32) -> tensor[2] f16:\n    let mut y = tensor[2] f16\n    y[0:2] = load(x)\n    return y\n");
        for name in ["assigned", "stored"] {
            let entry = module.entry(module.entry_named(name).unwrap(), &ElementBindings::default()).unwrap();
            let mut interpreter = Interpreter::new(&entry);
            let input = interpreter.add_tensor(TensorData::dense(DType::F32, vec![2], vec![1.0006, 2.0]));
            let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
            let result = outcome.results().next().unwrap();
            let OutcomeValue::Tensor(tensor) = result.value() else { panic!("{name} result") };
            assert_eq!(tensor.canonical_bytes().unwrap().unwrap(),
                [0x3c01u16, 0x4000].into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>(), "{name}");
        }
    }

    #[test]
    fn generic_tensor_installations_resolve_at_specialization() {
        let module = module("fn probe(x: tensor[64] U) -> tensor[64] T:\n    return x\n");
        let entry_id = module.entry_named("probe").unwrap();
        let packed = crate::registry::representation("q4g64").unwrap();
        let same_packed = ElementBindings::new().bind("U", packed).bind("T", packed);
        module.entry(entry_id, &same_packed).expect("equal packed representations install identically");
        let dense_floats = ElementBindings::new()
            .bind("U", crate::registry::dense(DType::F32))
            .bind("T", crate::registry::dense(DType::F16));
        let entry = module.entry(entry_id, &dense_floats).expect("dense float conversion is defined");
        let body = entry.program().function(entry.program().family(entry.program().root()).reference().function());
        assert!(body.nodes(body.root()).any(|(_, node)| matches!(node.view(),
            SemanticNodeView::Elementwise { primitive: PrimitiveId::Cast(DType::F16), .. })));
        let invalid = ElementBindings::new().bind("U", packed).bind("T", crate::registry::dense(DType::F32));
        let error = module.entry(entry_id, &invalid).unwrap_err();
        assert!(error.diagnostics().items().iter().any(|item| item.message.contains("no defined source installation conversion")));
    }

    #[test]
    fn generic_assignment_store_and_explicit_cast_use_concrete_rules() {
        let source = "fn assigned(x: tensor[64] T, y: tensor[64] U) -> tensor[64] T:\n    let mut z = x\n    z = y\n    return z\n\nfn stored(x: tensor[64] T, y: tensor[64] U) -> tensor[64] T:\n    let mut z = x\n    z[0:64] = y\n    return z\n\nfn casted(x: &tensor[64] U) -> tensor[64] f16:\n    return f16(x)\n\nfn decoded(x: &tensor[64] U) -> tensor[64] f32:\n    return f32(x)\n";
        let module = module(source);
        let packed = crate::registry::representation("q4g64").unwrap();
        let dense = ElementBindings::new()
            .bind("T", crate::registry::dense(DType::F16))
            .bind("U", crate::registry::dense(DType::F32));
        for name in ["assigned", "stored"] {
            module.entry(module.entry_named(name).unwrap(), &dense).expect("dense float installation");
        }
        let same_packed = ElementBindings::new().bind("T", packed).bind("U", packed);
        module.entry(module.entry_named("assigned").unwrap(), &same_packed).expect("equal packed whole assignment");
        let error = module.entry(module.entry_named("stored").unwrap(), &same_packed).unwrap_err();
        assert!(error.diagnostics().items().iter().any(|item| item.message.contains("decode-only")));
        let mixed = ElementBindings::new().bind("T", crate::registry::dense(DType::F32)).bind("U", packed);
        let error = module.entry(module.entry_named("assigned").unwrap(), &mixed).unwrap_err();
        assert!(error.diagnostics().items().iter().any(|item| item.message.contains("no defined source installation conversion")));
        let bound = ElementBindings::new().bind("U", packed);
        let error = module.entry(module.entry_named("casted").unwrap(), &bound).unwrap_err();
        assert!(error.diagnostics().items().iter().any(|item| item.message.contains("packed values decode with `f32(v)`")));
        module.entry(module.entry_named("decoded").unwrap(), &bound).expect("f32 packed decode is source-defined");
    }

    #[test]
    fn compound_integer_quantity_projects_to_source_word_first() {
        let module = module("fn probe(x: index[10000000000]) -> i32:\n    let mut y = i32(1)\n    y += x\n    return y\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let outcome = Interpreter::new(&entry).run(&[Arg::Index(4_294_967_297u64.into())]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(), OutcomeValue::Scalar(ReferenceScalar::I32(2))));
    }

    #[test]
    fn generic_packed_compound_target_has_no_implicit_encoding() {
        let module = module("fn probe(x: tensor[64] U) -> tensor[64] U:\n    let mut y = x\n    y += y\n    return y\n");
        let packed = crate::registry::representation("q4g64").unwrap();
        let bindings = ElementBindings::new().bind("U", packed);
        let error = module.entry(module.entry_named("probe").unwrap(), &bindings).unwrap_err();
        assert!(error.diagnostics().items().iter().any(|item| item.message.contains("no defined source installation conversion")));
    }

    #[test]
    fn mixed_narrow_float_compound_promotes_but_equal_f16_stays_f16() {
        let module = module("fn mixed(x: bf16) -> f16:\n    let mut y = f16(1.0)\n    y += x\n    return y\n\nfn equal(x: f16) -> f16:\n    let mut y = f16(1.0)\n    y += x\n    return y\n");
        for (name, argument, expected) in [
            ("mixed", ReferenceScalar::BF16(0x3a01), 0x3c01),
            ("equal", ReferenceScalar::F16(0x1000), 0x3c00),
        ] {
            let entry = module.entry(module.entry_named(name).unwrap(), &ElementBindings::default()).unwrap();
            let outcome = Interpreter::new(&entry).run(&[Arg::Scalar(argument)]).unwrap();
            assert!(matches!(outcome.results().next().unwrap().value(),
                OutcomeValue::Scalar(ReferenceScalar::F16(bits)) if bits == expected), "{name}");
        }
    }

    #[test]
    fn named_call_arguments_install_against_canonical_formals() {
        let module = module("fn helper(a: f16, b: f32) -> f16:\n    return a\n\nfn probe(x: f32, y: f32) -> f16:\n    return helper(b=y, a=x)\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let outcome = Interpreter::new(&entry).run(&[
            Arg::Scalar(ReferenceScalar::F32(0x3a00_0800)),
            Arg::Scalar(ReferenceScalar::F32(0x4000_0000)),
        ]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::F16(0x1000))));
    }

    #[test]
    fn nested_helper_dimension_comes_from_the_actual_sliced_argument() {
        let module = module("fn seed[M,K](x: &tensor[M,K] i32) -> i32:\n    return i32(M)\n\nfn probe(input: &tensor[8,2] i32, visible: &tensor[2] i32) -> i32:\n    let lo = visible[0]\n    let hi = visible[1]\n    let mut result = 0\n    if hi > lo:\n        result = seed(input[lo:hi,:])\n    return result\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter.add_tensor(TensorData::dense(DType::I32, vec![8, 2], vec![0.0; 16]));
        let visible = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![1.0, 4.0]));
        let outcome = interpreter.run(&[Arg::Tensor(input), Arg::Tensor(visible)]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(3))));
    }

    #[test]
    fn nested_helper_compound_dimension_inverts_actual_sliced_extent() {
        let module = module("fn seed[M,K](x: &tensor[M+1,K] i32) -> i32:\n    return i32(M)\n\nfn probe(input: &tensor[8,2] i32, visible: &tensor[2] i32) -> i32:\n    let lo = visible[0]\n    return seed(input[lo:lo+3,:])\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter.add_tensor(TensorData::dense(DType::I32, vec![8, 2], vec![0.0; 16]));
        let visible = interpreter.add_tensor(TensorData::dense(DType::I32, vec![2], vec![1.0, 4.0]));
        let outcome = interpreter.run(&[Arg::Tensor(input), Arg::Tensor(visible)]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(2))));
    }

    #[test]
    fn nested_helper_compound_dimension_uses_exact_division() {
        let module = module("fn seed[M](x: &tensor[2*M+1] i32) -> i32:\n    return i32(M)\n\nfn probe(input: &tensor[8] i32, visible: &tensor[1] i32) -> i32:\n    let lo = visible[0]\n    return seed(input[lo:lo+5])\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter.add_tensor(TensorData::dense(DType::I32, vec![8], vec![0.0; 8]));
        let visible = interpreter.add_tensor(TensorData::dense(DType::I32, vec![1], vec![1.0]));
        let outcome = interpreter.run(&[Arg::Tensor(input), Arg::Tensor(visible)]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(2))));
    }

    #[test]
    fn fixed_width_word_slice_checks_wrapped_endpoint_before_view() {
        let module = module("fn probe(input: &tensor[8] i32, lo: i32) -> i32:\n    let view = input[lo:lo+3]\n    return i32(extent(view,0))\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let run = |lo: i32| {
            let mut interpreter = Interpreter::new(&entry);
            let input = interpreter.add_tensor(TensorData::dense(DType::I32, vec![8], vec![0.0; 8]));
            interpreter.run(&[Arg::Tensor(input), Arg::Scalar(ReferenceScalar::I32(lo))]).unwrap()
        };
        let ok = run(1);
        assert!(matches!(ok.results().next().unwrap().value(), OutcomeValue::Scalar(ReferenceScalar::I32(3))));
        assert!(matches!(run(i32::MAX).termination(), crate::failure::SourceTermination::Failed(_)));
    }

    #[test]
    fn explicit_helper_dimension_uses_its_reached_source_value() {
        let module = module("fn seed[N](x: &tensor[N] i32) -> i32:\n    return i32(N)\n\nfn probe(input: &tensor[4] i32) -> i32:\n    return seed[N=extent(input,0)](input)\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter.add_tensor(TensorData::dense(DType::I32, vec![4], vec![0.0; 4]));
        let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(4))));
    }

    #[test]
    fn zero_extent_cannot_infer_negative_helper_dimension() {
        let source = "fn seed[M](x: &tensor[M+1] i32) -> i32:\n    return i32(M)\n\nfn probe(input: &tensor[4] i32) -> i32:\n    return seed(input[0:0])\n";
        assert!(check_source(SourceSet::new(vec![SourceFile {
            path: "negative-call-shape.seismic".into(), text: source.into(),
        }])).is_err());
    }

    #[test]
    fn zero_admitting_helper_dimension_accepts_empty_actual_view() {
        let module = module("fn seed[M](x: &tensor[M] i32) -> i32 where M >= 0:\n    return i32(M)\n\nfn probe(input: &tensor[4] i32) -> i32:\n    return seed(input[0:0])\n");
        let entry = module.entry(module.entry_named("probe").unwrap(), &ElementBindings::default()).unwrap();
        let mut interpreter = Interpreter::new(&entry);
        let input = interpreter.add_tensor(TensorData::dense(DType::I32, vec![4], vec![0.0; 4]));
        let outcome = interpreter.run(&[Arg::Tensor(input)]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(),
            OutcomeValue::Scalar(ReferenceScalar::I32(0))));
    }
}

use super::{ir, xfer};
use crate::checked::{internals::Module, SourceDiagnostic};
use crate::entry::*;
use crate::expr::{
    AnyExpr, BoolExpr, CmpOp, ExprArena, IntExpr, NodeView, ScalarArgument, ScalarComponent,
    SymbolId, SymbolSort, TargetPredicate,
};
use crate::ids::*;
use crate::intrinsics::PrimitiveId;
use crate::reference_math::ReferenceScalar;
use crate::syntax::ast::{AssignOp, BinaryOp};
use crate::types::{DType, Elem, ValueType};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) fn build_entry(
    module: &Module,
    entry: EntryId,
    bindings: &ElementBindings,
) -> Result<LogicalEntry, SourceDiagnostic> {
    let info = &module.entries[entry.index()];
    let family_ordinal = module.entry_families[entry.index()];
    let source_family = &module.families[family_ordinal];
    let contract = &module.definitions[source_family.contract.index()];
    let element_bindings = validate_elements(module, info, bindings)?;

    let program_id = ProgramId::fresh();
    let schema_id = CallSchema::fresh_id();
    let mut arena = ExprArena::new();
    let mut dimensions = Vec::new();
    let mut shape_values = Vec::new();
    for (ordinal, dimension) in contract.dimensions.iter().enumerate() {
        let id = CallSchema::dimension_id(schema_id, ordinal);
        let (symbol, value) = arena.call_dimension(id);
        dimensions.push(Dimension {
            id,
            name: dimension.name.clone(),
            symbol,
            admits_zero: dimension.admits_zero,
        });
        shape_values.push(arena.int_from_nat(value));
    }

    let mut builder = Builder::new(program_id, arena);
    let root_candidates = family_definitions(source_family)
        .into_iter()
        .map(|definition| CandidateSpec {
            definition,
            shape_args: shape_values.clone(),
            elements: element_bindings.clone(),
            // EntryDomain is exactly the root contract's precondition. Once
            // invocation validates that domain, the sealed portable contract
            // is universally applicable. Alternatives retain their own
            // structural target predicates.
            call_site_proved: definition == source_family.contract,
        })
        .collect();
    let root = instantiate_family(&mut builder, module, family_ordinal, root_candidates, false)?;

    let root_function = builder.families[root.index()]
        .as_ref()
        .and_then(|family| {
            family
                .candidates()
                .iter()
                .find(|candidate| candidate.numerical == NumericalRole::Reference)
        })
        .map(|candidate| candidate.function)
        .unwrap_or_else(|| panic!("checked entry family has no portable reference body"));
    let root_semantic = builder.functions[root_function.index()]
        .as_ref()
        .unwrap_or_else(|| panic!("root function slot was not populated"));

    let (parameters, aliases, domain_terms) = build_schema_parameters(
        &mut builder.arena,
        schema_id,
        contract,
        root_semantic,
        &shape_values,
        &element_bindings,
    );
    let results = build_schema_results(
        &mut builder.arena,
        contract,
        &shape_values,
        &element_bindings,
        root_semantic,
    );
    let dimension_inference =
        build_dimension_inference_plan(&builder.arena, &dimensions, &parameters)
            .map_err(|message| source_error(module, contract.file, contract.span, message))?;
    let schema = CallSchema::new(
        &builder.arena,
        schema_id,
        dimensions,
        dimension_inference,
        parameters,
        results,
        aliases,
    );
    let mut terms = domain_terms;
    for (dimension, value) in contract.dimensions.iter().zip(&shape_values) {
        if !dimension.admits_zero {
            let one = builder.arena.int(1);
            terms.push(builder.arena.int_cmp(CmpOp::Ge, *value, one));
        }
    }
    terms.extend(predicate_terms(&mut builder.arena, contract, &shape_values));
    let predicate = builder.arena.all(&terms);
    let totality = builder.arena.side_conditions(AnyExpr::Bool(predicate));
    let predicate = builder.arena.and(predicate, totality);
    let domain = EntryDomain::new(&builder.arena, predicate);

    let families = builder
        .families
        .into_iter()
        .map(|family| family.unwrap_or_else(|| panic!("semantic family slot was not populated")))
        .collect();
    let functions = builder
        .functions
        .into_iter()
        .map(|function| {
            function.unwrap_or_else(|| panic!("semantic function slot was not populated"))
        })
        .collect();
    let subject = std::sync::Arc::new(crate::entry::CheckedProgramSubject::new(
        std::sync::Arc::new(module.sources.clone()),
        element_bindings.into_iter().collect(),
    ));
    let program = SemanticProgram::new(crate::entry::internals::Program::new(
        program_id,
        subject,
        root,
        families,
        functions,
    ));
    Ok(LogicalEntry::new(
        info.stable,
        module.semantic_hash,
        schema,
        domain,
        builder.arena,
        program,
    ))
}

/// Build the one deterministic dimension solve plan for an external call.
/// All input axes are observations. In source order, repeatedly choose the
/// first axis expression containing exactly one unresolved dimension and
/// isolate that dimension through exact inverse operations. Other observed
/// axes may stand in for identical subexpressions, which permits deterministic
/// elimination such as observing both `N*G` and `2*N + N*G`.
fn build_dimension_inference_plan(
    arena: &ExprArena,
    dimensions: &[Dimension],
    parameters: &[Parameter],
) -> Result<DimensionInferencePlan, String> {
    let observations = parameters
        .iter()
        .flat_map(|parameter| match &parameter.kind {
            ParameterKind::Tensor { axes, .. } => axes.iter().copied().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();
    let dimension_specs = dimensions
        .iter()
        .map(|dimension| {
            (
                dimension.name.as_str(),
                dimension.symbol,
                dimension.admits_zero,
            )
        })
        .collect::<Vec<_>>();
    let observation_nodes = observations
        .iter()
        .copied()
        .map(AnyExpr::Nat)
        .collect::<Vec<_>>();
    let order = super::dimensions::dimension_inference_order(arena, &dimension_specs, &observation_nodes)
        .map_err(|underdetermined| {
            format!(
                "external entry dimensions {} are underdetermined or require a nonlinear/invocation-dependent inversion; every dimension must be uniquely derivable from input tensor extents",
                underdetermined.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ")
            )
        })?;
    let steps = order
        .into_iter()
        .map(|(dimension, observation, operations)| {
            let operations = operations
                .into_iter()
                .map(|operation| match operation {
                    super::dimensions::InferenceOp::Add(known) => DimensionInferenceOp::Add(final_known(known)),
                    super::dimensions::InferenceOp::Subtract(known) => {
                        DimensionInferenceOp::Subtract(final_known(known))
                    }
                    super::dimensions::InferenceOp::DivideExact(known) => {
                        DimensionInferenceOp::DivideExact(final_known(known))
                    }
                    super::dimensions::InferenceOp::ReverseSubtract(known) => {
                        DimensionInferenceOp::ReverseSubtract(final_known(known))
                    }
                })
                .collect();
            (
                dimension,
                observation,
                observations[observation],
                operations,
            )
        })
        .collect();

    Ok(DimensionInferencePlan::new(observations.len(), steps))
}

fn final_known(known: super::dimensions::InferenceKnown) -> DimensionInferenceKnown {
    match known {
        super::dimensions::InferenceKnown::Observation(observation) => {
            DimensionInferenceKnown::Observation(observation)
        }
        super::dimensions::InferenceKnown::Expression(AnyExpr::Nat(expression)) => {
            DimensionInferenceKnown::Nat(expression)
        }
        super::dimensions::InferenceKnown::Expression(AnyExpr::Int(expression)) => {
            DimensionInferenceKnown::Int(expression)
        }
        super::dimensions::InferenceKnown::Expression(_) => {
            panic!("dimension inference produced a non-integer known expression")
        }
    }
}

fn validate_elements(
    module: &Module,
    info: &crate::checked::EntryInfo,
    supplied: &ElementBindings,
) -> Result<BTreeMap<String, RepresentationId>, SourceDiagnostic> {
    let expected: BTreeSet<_> = info.element_parameters.iter().cloned().collect();
    let actual: BTreeSet<_> = supplied.iter().map(|(name, _)| name.to_owned()).collect();
    if expected != actual {
        let missing: Vec<_> = expected.difference(&actual).cloned().collect();
        let extra: Vec<_> = actual.difference(&expected).cloned().collect();
        return Err(SourceDiagnostic::new(
            &module.sources.files()[0],
            crate::span::Span::default(),
            crate::checked::DiagnosticRule::Type,
            format!(
                "element bindings do not match the entry parameters; missing: {missing:?}; extra: {extra:?}"
            ),
        ));
    }
    Ok(supplied
        .iter()
        .map(|(name, representation)| (name.to_owned(), representation))
        .collect())
}

fn family_definitions(family: &ir::Family) -> Vec<FunctionId> {
    family
        .bodies
        .iter()
        .chain(&family.lowerings)
        .copied()
        .collect()
}

#[derive(Clone)]
struct CandidateSpec {
    definition: FunctionId,
    shape_args: Vec<IntExpr>,
    elements: BTreeMap<String, RepresentationId>,
    /// Nested calls have already proved their contract precondition in the
    /// caller. Root candidates retain their structural applicability.
    call_site_proved: bool,
}

struct Builder {
    program: ProgramId,
    arena: ExprArena,
    families: Vec<Option<Family>>,
    functions: Vec<Option<SemanticFunction>>,
}

impl Builder {
    fn new(program: ProgramId, arena: ExprArena) -> Self {
        Self {
            program,
            arena,
            families: Vec::new(),
            functions: Vec::new(),
        }
    }
    fn reserve_family(&mut self) -> FamilyId {
        let id = FamilyId::new(
            self.program,
            u32::try_from(self.families.len()).expect("entry has more than u32::MAX families"),
        );
        self.families.push(None);
        id
    }
    fn reserve_function(&mut self) -> FunctionId {
        let id = FunctionId::new(
            self.program,
            u32::try_from(self.functions.len()).expect("entry has more than u32::MAX functions"),
        );
        self.functions.push(None);
        id
    }
}

fn instantiate_family(
    builder: &mut Builder,
    module: &Module,
    source_family: usize,
    specs: Vec<CandidateSpec>,
    capture_dimensions: bool,
) -> Result<FamilyId, SourceDiagnostic> {
    let id = builder.reserve_family();
    let family = &module.families[source_family];
    let mut candidates = Vec::new();
    let mut reference_assigned = false;
    for spec in specs {
        let definition = &module.definitions[spec.definition.index()];
        let Some(elements) = specialize_elements(definition, &spec.elements) else {
            continue;
        };
        let applicability = if spec.call_site_proved {
            let yes = builder.arena.bool(true);
            TargetPredicate::new(&builder.arena, yes)
                .unwrap_or_else(|_| panic!("constant applicability was rejected"))
        } else {
            let Some(applicability) = applicability(builder, definition, &spec.shape_args) else {
                // A specialized alternative whose predicate depends on a
                // loop/data value cannot participate in target planning.
                // The portable contract remains available; source may guard
                // and call a separately named helper explicitly.
                continue;
            };
            applicability
        };
        let function = instantiate_function(
            builder,
            module,
            spec.definition.index(),
            spec.shape_args,
            elements,
            capture_dimensions,
        )?;
        let kind = match definition.kind {
            ir::DefKind::Body { target: None } => CandidateKind::Portable,
            ir::DefKind::Body {
                target: Some(backend),
            } => CandidateKind::Helper { backend },
            ir::DefKind::Lower { target } => CandidateKind::Lowering { backend: target },
        };
        let numerical = if spec.definition == family.contract {
            assert!(
                definition.kind.is_portable_body(),
                "checked family contract is not a portable body"
            );
            assert!(
                !reference_assigned,
                "family has more than one reference body"
            );
            reference_assigned = true;
            NumericalRole::Reference
        } else {
            NumericalRole::Alternative
        };
        candidates.push(Candidate {
            function,
            kind,
            requires: definition.requires.clone(),
            applicability,
            numerical,
        });
    }
    if candidates.is_empty() {
        return Err(source_error(
            module,
            module.definitions[family.contract.index()].file,
            module.definitions[family.contract.index()].span,
            "no implementation applies to these element bindings",
        ));
    }
    if !reference_assigned {
        return Err(source_error(
            module,
            module.definitions[family.contract.index()].file,
            module.definitions[family.contract.index()].span,
            "no portable reference implementation applies to these element bindings",
        ));
    }
    builder.families[id.index()] = Some(Family::new(family.name.clone(), candidates));
    Ok(id)
}

fn specialize_elements(
    definition: &ir::Definition,
    inherited: &BTreeMap<String, RepresentationId>,
) -> Option<BTreeMap<String, RepresentationId>> {
    let mut out = inherited.clone();
    for (name, element) in &definition.elem_bindings {
        let concrete = match element {
            Elem::Dtype(dtype) => crate::registry::dense(*dtype),
            Elem::Repr(representation) => *representation,
            Elem::Param(parameter) => *out.get(parameter)?,
        };
        if out.get(name).is_some_and(|existing| *existing != concrete) {
            return None;
        }
        out.insert(name.clone(), concrete);
    }
    Some(out)
}

fn applicability(
    builder: &mut Builder,
    definition: &ir::Definition,
    shapes: &[IntExpr],
) -> Option<TargetPredicate> {
    let mut terms = Vec::new();
    for (ordinal, dimension) in definition.dimensions.iter().enumerate() {
        let lower = builder.arena.int(if dimension.admits_zero { 0 } else { 1 });
        terms.push(builder.arena.int_cmp(CmpOp::Ge, shapes[ordinal], lower));
    }
    for predicate in &definition.predicates {
        let expression = transfer_template_int(
            &mut builder.arena,
            definition,
            shapes,
            predicate.expression(),
        );
        let zero = builder.arena.int(0);
        terms.push(match predicate {
            ir::Predicate::NonNegative(_) => builder.arena.int_cmp(CmpOp::Ge, expression, zero),
            ir::Predicate::Zero(_) => builder.arena.int_cmp(CmpOp::Eq, expression, zero),
            ir::Predicate::NonZero(_) => builder.arena.int_cmp(CmpOp::Ne, expression, zero),
        });
    }
    let predicate = builder.arena.all(&terms);
    TargetPredicate::new(&builder.arena, predicate).ok()
}

trait PredicateExpr {
    fn expression(&self) -> IntExpr;
}
impl PredicateExpr for ir::Predicate {
    fn expression(&self) -> IntExpr {
        match *self {
            Self::NonNegative(e) | Self::Zero(e) | Self::NonZero(e) => e,
        }
    }
}

fn transfer_template_int(
    arena: &mut ExprArena,
    definition: &ir::Definition,
    shapes: &[IntExpr],
    expression: IntExpr,
) -> IntExpr {
    let mut map = |symbol: SymbolId, _: &mut ExprArena| {
        let ordinal = definition
            .dimensions
            .iter()
            .position(|dimension| dimension.symbol == symbol)
            .unwrap_or_else(|| panic!("template expression mentions a non-dimension symbol"));
        AnyExpr::Int(shapes[ordinal])
    };
    xfer::transfer_int(&definition.arena, expression, arena, &mut map)
}

fn source_error(
    module: &Module,
    file: usize,
    span: crate::span::Span,
    message: impl Into<String>,
) -> SourceDiagnostic {
    SourceDiagnostic::new(
        &module.sources.files()[file],
        span,
        crate::checked::DiagnosticRule::Type,
        message.into(),
    )
}

fn build_schema_parameters(
    arena: &mut ExprArena,
    schema: SchemaId,
    definition: &ir::Definition,
    function: &SemanticFunction,
    _shapes: &[IntExpr],
    _elements: &BTreeMap<String, RepresentationId>,
) -> (Vec<Parameter>, Vec<AliasRule>, Vec<BoolExpr>) {
    let mut parameters = Vec::new();
    let mut domain = Vec::new();
    for (ordinal, semantic) in function.parameters().iter().enumerate() {
        let FunctionParameterOrigin::Source { ordinal: source, path } = &semantic.origin else {
            panic!("public entry schema contains a captured helper dimension")
        };
        let id = CallSchema::parameter_id(schema, ordinal);
        let ty = function.value(semantic.value).ty.clone();
        let kind = match ty {
            SemanticType::Tensor(tensor) => ParameterKind::Tensor {
                access: match semantic.access {
                    ParameterAccess::Owned => TensorAccess::Owned,
                    ParameterAccess::Shared => TensorAccess::Shared,
                    ParameterAccess::Mutable => TensorAccess::Mutable,
                    ParameterAccess::Scalar => {
                        panic!("checked tensor parameter has scalar access")
                    }
                },
                representation: tensor.representation,
                axes: tensor.axes,
            },
            SemanticType::Scalar(dtype) => {
                let symbol = arena.call_scalar(
                    ScalarArgument { parameter: id, component: ScalarComponent::Value },
                    scalar_sort(dtype),
                );
                ParameterKind::Scalar { dtype, symbol }
            }
            SemanticType::Index { bound } => {
                let symbol = arena.call_scalar(
                    ScalarArgument { parameter: id, component: ScalarComponent::Value },
                    SymbolSort::Nat,
                );
                let value = arena.nat_symbol(symbol);
                domain.push(arena.nat_cmp(CmpOp::Lt, value, bound));
                ParameterKind::Index { bound, symbol }
            }
            SemanticType::Range { bound } => {
                let start = arena.call_scalar(
                    ScalarArgument { parameter: id, component: ScalarComponent::RangeStart },
                    SymbolSort::Nat,
                );
                let end = arena.call_scalar(
                    ScalarArgument { parameter: id, component: ScalarComponent::RangeEnd },
                    SymbolSort::Nat,
                );
                let start_value = arena.nat_symbol(start);
                let end_value = arena.nat_symbol(end);
                domain.push(arena.nat_cmp(CmpOp::Ge, end_value, start_value));
                domain.push(arena.nat_cmp(CmpOp::Le, end_value, bound));
                ParameterKind::Range { bound, start, end }
            }
            SemanticType::Integer
            | SemanticType::Tuple(_)
            | SemanticType::Opaque { .. }
            | SemanticType::Void => {
                panic!("unsupported checked entry parameter escaped signature checking")
            }
        };
        parameters.push(Parameter {
            id,
            source: *source,
            path: path.clone(),
            name: semantic.name.clone(),
            kind,
            value: semantic.value,
        });
    }
    let aliases = definition
        .aliases
        .iter()
        .flat_map(|(left, right)| {
            let left = function
                .parameters()
                .iter()
                .enumerate()
                .filter_map(move |(ordinal, parameter)| {
                    matches!(&parameter.origin, FunctionParameterOrigin::Source { ordinal: source, .. } if *source as usize == *left).then_some(ordinal)
                })
                .collect::<Vec<_>>();
            let right = function
                .parameters()
                .iter()
                .enumerate()
                .filter_map(move |(ordinal, parameter)| {
                    matches!(&parameter.origin, FunctionParameterOrigin::Source { ordinal: source, .. } if *source as usize == *right).then_some(ordinal)
                })
                .collect::<Vec<_>>();
            left.into_iter()
                .flat_map(move |left| right.clone().into_iter().map(move |right| (left, right)))
        })
        .map(|(left, right)| AliasRule::MayOverlap(parameters[left].id, parameters[right].id))
        .collect();
    // All tensor pairs not explicitly admitted to alias are disjoint whenever
    // either side is mutable/owned. Shared/shared pairs may overlap.
    let mut aliases: Vec<AliasRule> = aliases;
    for left in 0..parameters.len() {
        for right in left + 1..parameters.len() {
            let FunctionParameterOrigin::Source { ordinal: left_source, .. } = &function.parameters()[left].origin else { unreachable!() };
            let FunctionParameterOrigin::Source { ordinal: right_source, .. } = &function.parameters()[right].origin else { unreachable!() };
            let left_source = *left_source as usize;
            let right_source = *right_source as usize;
            if definition.aliases.iter().any(|(a, b)| {
                (*a == left_source && *b == right_source)
                    || (*a == right_source && *b == left_source)
            }) {
                continue;
            }
            let tensor =
                |parameter: &Parameter| matches!(parameter.kind, ParameterKind::Tensor { .. });
            let shared = |parameter: &Parameter| {
                matches!(
                    parameter.kind,
                    ParameterKind::Tensor {
                        access: TensorAccess::Shared,
                        ..
                    }
                )
            };
            if tensor(&parameters[left]) && tensor(&parameters[right]) {
                aliases.push(if shared(&parameters[left]) && shared(&parameters[right]) {
                    AliasRule::MayOverlap(parameters[left].id, parameters[right].id)
                } else {
                    AliasRule::Disjoint(parameters[left].id, parameters[right].id)
                });
            }
        }
    }
    (parameters, aliases, domain)
}

fn scalar_sort(dtype: DType) -> SymbolSort {
    // ABI scalar symbols retain the source word sort. Geometry projects the
    // actual word through IntFromScalar instead of relabeling it as Int/Nat.
    SymbolSort::Scalar(dtype)
}

fn build_schema_results(
    arena: &mut ExprArena,
    definition: &ir::Definition,
    shapes: &[IntExpr],
    elements: &BTreeMap<String, RepresentationId>,
    function: &SemanticFunction,
) -> Vec<ResultLeaf> {
    let declared = super::result_leaves(&definition.result);
    assert_eq!(declared.len(), function.results().len(), "checked result leaf count changed during semantic lowering");
    declared.into_iter().zip(function.results().iter().copied()).map(|((path, ty), value)| {
        let kind = match ty {
            ValueType::Tensor(tensor) => {
                let representation = match &tensor.elem {
                    Elem::Dtype(dtype) => crate::registry::dense(*dtype),
                    Elem::Repr(representation) => *representation,
                    Elem::Param(name) => *elements.get(name).expect("entry result element parameter was not bound"),
                };
                let axes = tensor.axes.iter().map(|axis| {
                    let expression = transfer_template_int(arena, definition, shapes, *axis);
                    arena.nat_from_int(expression)
                }).collect::<Vec<_>>();
                let SemanticType::Tensor(actual) = &function.value(value).ty else {
                    panic!("checked tensor result has a non-tensor semantic producer")
                };
                assert_eq!(actual.representation, representation, "return conversion changed declared representation");
                assert_eq!(actual.axes.len(), axes.len(), "return conversion changed declared rank");
                ResultKind::Tensor { representation, axes }
            }
            ValueType::Scalar(dtype) => {
                assert!(matches!(function.value(value).ty, SemanticType::Scalar(actual) if actual == *dtype), "return conversion changed declared scalar dtype");
                ResultKind::Scalar(*dtype)
            }
            ValueType::Index { bound } => {
                assert!(matches!(function.value(value).ty, SemanticType::Index { .. }), "checked index result has a non-index semantic producer");
                let expression = transfer_template_int(arena, definition, shapes, *bound);
                ResultKind::Index { bound: arena.nat_from_int(expression) }
            }
            ValueType::Range { bound } => {
                assert!(matches!(function.value(value).ty, SemanticType::Range { .. }), "checked range result has a non-range semantic producer");
                let expression = transfer_template_int(arena, definition, shapes, *bound);
                ResultKind::Range { bound: arena.nat_from_int(expression) }
            }
            _ => unreachable!("result leaves contain only public value types"),
        };
        ResultLeaf { path, kind, value }
    }).collect()
}

fn predicate_terms(
    arena: &mut ExprArena,
    definition: &ir::Definition,
    shapes: &[IntExpr],
) -> Vec<BoolExpr> {
    definition
        .predicates
        .iter()
        .map(|predicate| {
            let expression =
                transfer_template_int(arena, definition, shapes, predicate.expression());
            let zero = arena.int(0);
            match predicate {
                ir::Predicate::NonNegative(_) => arena.int_cmp(CmpOp::Ge, expression, zero),
                ir::Predicate::Zero(_) => arena.int_cmp(CmpOp::Eq, expression, zero),
                ir::Predicate::NonZero(_) => arena.int_cmp(CmpOp::Ne, expression, zero),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Checked-template -> entry-owned semantic program
// ---------------------------------------------------------------------------

fn instantiate_function(
    builder: &mut Builder,
    module: &Module,
    source: usize,
    shapes: Vec<IntExpr>,
    elements: BTreeMap<String, RepresentationId>,
    capture_dimensions: bool,
) -> Result<FunctionId, SourceDiagnostic> {
    let definition = &module.definitions[source];
    let id = builder.reserve_function();
    let source_definition = u64::try_from(source).expect("source definition ordinal exceeds u64");
    let mut lowering = FunctionLowering::new(
        builder,
        module,
        definition,
        source_definition,
        id,
        shapes,
        elements,
        capture_dimensions,
    );
    let function = lowering.lower()?;
    drop(lowering);
    builder.functions[id.index()] = Some(function);
    Ok(id)
}

struct RegionWork {
    id: RegionId,
    kind: RegionKind,
    parameters: Vec<SemanticValueId>,
    results: Vec<SemanticValueId>,
    nodes: Vec<SemanticNode>,
    initialization: Option<crate::initialization::LoopInitialization>,
}

struct FunctionLowering<'a, 'm> {
    builder: &'a mut Builder,
    module: &'m Module,
    definition: &'m ir::Definition,
    source_definition: u64,
    id: FunctionId,
    shapes: Vec<IntExpr>,
    capture_dimensions: bool,
    elements: BTreeMap<String, RepresentationId>,
    values: Vec<ValueInfo>,
    regions: Vec<Option<RegionWork>>,
    locals: Vec<Option<SemanticValueId>>,
    /// Evaluated formals remain immutable when their source local is reassigned.
    formals: Vec<SemanticValueId>,
    symbols: HashMap<SymbolId, IntExpr>,
    runtime_values: HashMap<SemanticValueId, IntExpr>,
    parameters: Vec<FunctionParameter>,
    parallel_depth: u32,
    parallel_participants: Vec<BinderId>,
    participant_bindings: HashMap<ir::LocalId, BinderId>,
    last_event: HashMap<RegionId, Vec<SemanticEventId>>,
}

impl<'a, 'm> FunctionLowering<'a, 'm> {
    fn new(
        builder: &'a mut Builder,
        module: &'m Module,
        definition: &'m ir::Definition,
        source_definition: u64,
        id: FunctionId,
        shapes: Vec<IntExpr>,
        elements: BTreeMap<String, RepresentationId>,
        capture_dimensions: bool,
    ) -> Self {
        let mut symbols = HashMap::new();
        for (dimension, value) in definition.dimensions.iter().zip(&shapes) {
            symbols.insert(dimension.symbol, *value);
        }
        Self {
            builder,
            module,
            definition,
            source_definition,
            id,
            shapes,
            capture_dimensions,
            elements,
            values: Vec::new(),
            regions: Vec::new(),
            locals: vec![None; definition.body.locals.len()],
            formals: Vec::new(),
            symbols,
            runtime_values: HashMap::new(),
            parameters: Vec::new(),
            parallel_depth: 0,
            parallel_participants: Vec::new(),
            participant_bindings: HashMap::new(),
            last_event: HashMap::new(),
        }
    }

    fn lower(&mut self) -> Result<SemanticFunction, SourceDiagnostic> {
        let root = self.reserve_region(RegionKind::Root);
        if self.capture_dimensions {
            let dimensions = self.definition.dimensions.clone();
            for (ordinal, dimension) in dimensions.iter().enumerate() {
                let value = self.value(SemanticType::Integer, ValueOrigin::Parameter, self.definition.span);
                self.parameters.push(FunctionParameter {
                    origin: FunctionParameterOrigin::ShapeDimension {
                        ordinal: u32::try_from(ordinal).expect("function has more than u32::MAX dimensions"),
                    },
                    name: dimension.name.clone(),
                    value,
                    access: ParameterAccess::Scalar,
                });
                let actual = self.runtime_int(value);
                self.shapes[ordinal] = actual;
                self.symbols.insert(dimension.symbol, actual);
            }
        }
        for (source, parameter) in self.definition.params.clone().into_iter().enumerate() {
            let ty = self.semantic_type(&parameter.ty);
            let value = self.lower_parameter(
                root,
                u32::try_from(source).expect("function has more than u32::MAX parameters"),
                &parameter.name,
                &[],
                ty,
                &parameter.ownership,
                parameter.span,
            )?;
            self.locals[parameter.local.index()] = Some(value);
            if let Some(source_symbol) = self.definition.body.locals[parameter.local.index()].symbol
            {
                let runtime = self.runtime_int(value);
                self.symbols.insert(source_symbol, runtime);
            }
        }
        self.formals = self
            .parameters
            .iter()
            .map(|parameter| parameter.value)
            .collect();
        let block = self.definition.body.root.clone();
        let results = self.lower_block(root, &block)?;
        let declared = super::result_leaves(&self.definition.result)
            .into_iter()
            .map(|(_, ty)| ty.clone())
            .collect::<Vec<_>>();
        assert_eq!(results.len(), declared.len(), "checked return changed its declared leaf count");
        let results = results.into_iter().zip(declared).map(|(value, ty)| {
            let expected = self.semantic_type(&ty);
            self.install_value(root, value, &expected, self.values[value.index()].span)
        }).collect::<Result<Vec<_>, _>>()?;
        let regions = std::mem::take(&mut self.regions)
            .into_iter()
            .map(|region| {
                let region =
                    region.unwrap_or_else(|| panic!("semantic region slot was not populated"));
                Region::new(
                    region.kind,
                    region.parameters,
                    region.results,
                    region.nodes,
                    region.initialization,
                )
            })
            .collect();
        let parameters = std::mem::take(&mut self.parameters);
        let leaves = parameters
            .iter()
            .enumerate()
            .filter_map(|(ordinal, p)| {
                let FunctionParameterOrigin::Source { ordinal: source, path } = &p.origin else { return None };
                Some((*source as usize,
                    path.iter().map(|p| *p as usize).collect(), ordinal))
            })
            .collect::<Vec<_>>();
        let symbols = &self.symbols;
        let initialization = self.definition.initialization.remap(
            &self.definition.arena,
            &mut self.builder.arena,
            &leaves,
            &mut |symbol, _| {
                AnyExpr::Int(
                    *symbols
                        .get(&symbol)
                        .expect("closed initialization symbol is a shape or formal parameter"),
                )
            },
        );
        let values = std::mem::take(&mut self.values);
        Ok(SemanticFunction::new(
            crate::entry::internals::Function::new(
                self.id,
                self.definition.stable,
                self.source_definition,
                self.definition.name.clone(),
                self.definition.span,
                parameters,
                results,
                root,
                regions,
                values,
                initialization,
            ),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_parameter(
        &mut self,
        root: RegionId,
        source: u32,
        name: &str,
        path: &[u32],
        ty: SemanticType,
        ownership: &ir::Ownership,
        span: crate::span::Span,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        if let SemanticType::Tuple(items) = &ty {
            let mut inputs = Vec::with_capacity(items.len());
            for (ordinal, item) in items.iter().cloned().enumerate() {
                let mut child = path.to_vec();
                child.push(u32::try_from(ordinal).expect("tuple has more than u32::MAX items"));
                let ir::Ownership::Tuple(parts) = ownership else {
                    panic!("checked tuple parameter lost ownership")
                };
                inputs.push(self.lower_parameter(
                    root,
                    source,
                    name,
                    &child,
                    item,
                    &parts[ordinal],
                    span,
                )?);
            }
            return Ok(self.emit(root, NodeKind::TuplePack, inputs, vec![ty], span)[0]);
        }
        let leaf_access = match ownership {
            ir::Ownership::Owned => ParameterAccess::Owned,
            ir::Ownership::Shared => ParameterAccess::Shared,
            ir::Ownership::Exclusive => ParameterAccess::Mutable,
            ir::Ownership::Value => ParameterAccess::Scalar,
            ir::Ownership::Tuple(_) => panic!("non-tuple parameter has tuple ownership"),
        };
        let mut ty = ty;
        if let SemanticType::Tensor(tensor) = &mut ty {
            if leaf_access == ParameterAccess::Mutable
                && crate::registry::representation_info(tensor.representation).access
                    != crate::registry::RepresentationAccess::ReadWrite
            {
                return Err(SourceDiagnostic::new(
                    &self.module.sources.files()[self.definition.file],
                    span,
                    crate::checked::DiagnosticRule::Type,
                    format!(
                        "representation `{}` is decode-only and cannot bind a mutable tensor parameter",
                        crate::registry::representation_info(tensor.representation).name
                    ),
                ));
            }
            tensor.storage = TensorStorage::Parameter(leaf_access);
        }
        let value = self.value(ty, ValueOrigin::Parameter, span);
        let mut leaf_name = name.to_owned();
        for ordinal in path {
            leaf_name.push('_');
            leaf_name.push_str(&ordinal.to_string());
        }
        self.parameters.push(FunctionParameter {
            origin: FunctionParameterOrigin::Source { ordinal: source, path: path.to_vec() },
            name: leaf_name,
            value,
            access: leaf_access,
        });
        Ok(value)
    }

    fn reserve_region(&mut self, kind: RegionKind) -> RegionId {
        let id = RegionId::new(
            self.id,
            u32::try_from(self.regions.len()).expect("function has more than u32::MAX regions"),
        );
        self.regions.push(Some(RegionWork {
            id,
            kind,
            parameters: Vec::new(),
            results: Vec::new(),
            nodes: Vec::new(),
            initialization: None,
        }));
        id
    }

    fn value(
        &mut self,
        ty: SemanticType,
        origin: ValueOrigin,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let id = SemanticValueId::new(
            self.id,
            u32::try_from(self.values.len()).expect("function has more than u32::MAX values"),
        );
        self.values.push(ValueInfo { ty, origin, span });
        id
    }

    fn region_parameter(
        &mut self,
        region: RegionId,
        ty: SemanticType,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let value = self.value(ty, ValueOrigin::RegionParameter(region), span);
        self.region_mut(region).parameters.push(value);
        value
    }

    fn region_mut(&mut self, id: RegionId) -> &mut RegionWork {
        assert_eq!(
            id.function(),
            self.id,
            "region belongs to another semantic function"
        );
        self.regions[id.index()]
            .as_mut()
            .unwrap_or_else(|| panic!("semantic region slot is absent"))
    }

    fn emit(
        &mut self,
        region: RegionId,
        kind: NodeKind,
        inputs: Vec<SemanticValueId>,
        output_types: Vec<SemanticType>,
        span: crate::span::Span,
    ) -> Vec<SemanticValueId> {
        let ordinal = self.region_mut(region).nodes.len();
        let node = NodeId::new(
            region,
            u32::try_from(ordinal).expect("region has more than u32::MAX nodes"),
        );
        let outputs = output_types
            .into_iter()
            .map(|ty| self.value(ty, ValueOrigin::Node(node), span))
            .collect::<Vec<_>>();
        let events = self.semantic_events(node, &kind, &inputs, &outputs);
        self.region_mut(region).nodes.push(SemanticNode::new(
            kind,
            inputs,
            outputs.clone(),
            events,
            span,
        ));
        outputs
    }

    fn participant_domain(&self) -> ParticipantDomain {
        if self.parallel_participants.is_empty() {
            ParticipantDomain::Single
        } else {
            ParticipantDomain::Parallel(self.parallel_participants.clone().into_boxed_slice())
        }
    }

    fn tensor_representation(&self, value: SemanticValueId) -> Option<RepresentationId> {
        match &self.values[value.index()].ty {
            SemanticType::Tensor(tensor) => Some(tensor.representation),
            _ => None,
        }
    }

    fn semantic_events(
        &mut self,
        node: NodeId,
        kind: &NodeKind,
        inputs: &[SemanticValueId],
        outputs: &[SemanticValueId],
    ) -> Vec<SemanticEvent> {
        let participants = self.participant_domain();
        let mut specs = Vec::new();
        let mut may_fail = match kind {
            NodeKind::Check { .. } => true,
            NodeKind::Primitive(primitive) | NodeKind::Elementwise(primitive) => {
                matches!(primitive, PrimitiveId::Binary(crate::syntax::ast::BinaryOp::Div | crate::syntax::ast::BinaryOp::Rem))
                    && outputs.iter().any(|id| matches!(self.values[id.index()].ty, SemanticType::Integer))
                    || semantic_scalar_recipe(
                        primitive,
                        inputs.iter().map(|id| &self.values[id.index()].ty),
                    )
                    .is_some_and(|recipe| !recipe.failures().is_empty())
            }
            _ => false,
        };
        match kind {
            NodeKind::Atomic { op, capability } => {
                let place = *inputs
                    .first()
                    .unwrap_or_else(|| panic!("checked atomic has no write place"));
                let representation = self
                    .tensor_representation(place)
                    .unwrap_or_else(|| panic!("checked atomic place is not a tensor"));
                let dtype = crate::registry::representation_info(representation).decoded;
                let outcome = match op {
                    crate::intrinsics::AtomicOp::Add
                        if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) =>
                    {
                        AssociationOutcome::Reassociated { accumulator: dtype }
                    }
                    crate::intrinsics::AtomicOp::Add
                    | crate::intrinsics::AtomicOp::Max
                    | crate::intrinsics::AtomicOp::Min => AssociationOutcome::Exact,
                };
                assert_eq!(capability.place(), place);
                assert_eq!(capability.outcome(), outcome);
                let id = SemanticEventId::new(node, 0);
                let event = SemanticEvent::checked(
                    id,
                    Some(place),
                    capability.indices().to_vec(),
                    SemanticEventKind::AtomicRmw {
                        op: *op,
                        capability: capability.clone(),
                    },
                    Some(representation),
                    capability.participants().clone(),
                    self.last_event
                        .get(&node.region())
                        .cloned()
                        .unwrap_or_default(),
                    VisibilityScope::Command,
                    match outcome {
                        AssociationOutcome::Exact => NumericalOutcome::Deterministic,
                        outcome => NumericalOutcome::AllowedAssociation(outcome),
                    },
                );
                self.last_event.insert(node.region(), vec![id]);
                return vec![event];
            }
            NodeKind::If { then, otherwise } => {
                // Child operations own their events. Control nodes never
                // duplicate those events as a second semantic authority.
                let mut terminal = self.last_event.get(then).cloned().unwrap_or_default();
                for event in self.last_event.get(otherwise).into_iter().flatten() {
                    if !terminal.contains(event) {
                        terminal.push(*event);
                    }
                }
                self.last_event.insert(node.region(), terminal);
            }
            NodeKind::Loop { body, .. } => {
                // Include the pre-loop predecessor for the zero-visit path
                // and the body terminal for every nonempty execution.
                let mut terminal = self
                    .last_event
                    .get(&node.region())
                    .cloned()
                    .unwrap_or_default();
                for event in self.last_event.get(body).into_iter().flatten() {
                    if !terminal.contains(event) {
                        terminal.push(*event);
                    }
                }
                self.last_event.insert(node.region(), terminal);
            }
            NodeKind::Alloc | NodeKind::Fill { .. } => {
                for output in outputs {
                    if let Some(representation) = self.tensor_representation(*output) {
                        specs.push((
                            Some(*output),
                            SemanticEventKind::Write(None),
                            Some(representation),
                            VisibilityScope::Participant,
                            NumericalOutcome::Deterministic,
                        ));
                    }
                }
            }
            NodeKind::Primitive(_)
            | NodeKind::View(_)
            | NodeKind::Check { .. }
            | NodeKind::TuplePack
            | NodeKind::TupleGet { .. }
            | NodeKind::Extent { .. } => {}
            NodeKind::Intrinsic(intrinsic) => {
                let signature = crate::registry::intrinsic_signature(*intrinsic);
                let reads =
                    signature
                        .arguments
                        .iter()
                        .enumerate()
                        .filter_map(|(ordinal, argument)| {
                            matches!(
                                argument.category,
                                crate::registry::OperandCategory::Readable { .. }
                                    | crate::registry::OperandCategory::Writable { .. }
                            )
                            .then_some(inputs[ordinal])
                        });
                let writes = signature.effects.writes.iter().map(|ordinal| {
                    inputs[usize::try_from(*ordinal).expect("intrinsic write index exceeds usize")]
                });
                Self::push_access_specs(self, &mut specs, reads, writes);
                Self::push_owned_output_specs(self, &mut specs, outputs);
            }
            NodeKind::Elementwise(_) => {
                let reads = inputs
                    .iter()
                    .copied()
                    .filter(|value| self.tensor_representation(*value).is_some())
                    .collect::<Vec<_>>();
                Self::push_access_specs(self, &mut specs, reads, std::iter::empty());
            }
            NodeKind::Reduce { .. } | NodeKind::ElementRead => {
                Self::push_access_specs(
                    self,
                    &mut specs,
                    inputs.first().copied(),
                    std::iter::empty(),
                );
            }
            NodeKind::Call { family } => {
                let reference = self.builder.families[family.index()]
                    .as_ref()
                    .and_then(|family| {
                        family
                            .candidates()
                            .iter()
                            .find(|candidate| candidate.numerical == NumericalRole::Reference)
                    })
                    .map(|candidate| candidate.function)
                    .unwrap_or_else(|| panic!("checked call family has no reference function"));
                let function = self.builder.functions[reference.index()]
                    .as_ref()
                    .unwrap_or_else(|| panic!("checked call reference function is absent"));
                may_fail = function.has_failure_events();
                let reads = function
                    .parameters()
                    .iter()
                    .zip(inputs)
                    .filter_map(|(parameter, input)| {
                        matches!(
                            parameter.access,
                            ParameterAccess::Shared
                                | ParameterAccess::Owned
                                | ParameterAccess::Mutable
                        )
                        .then_some(*input)
                    })
                    .collect::<Vec<_>>();
                let writes = function
                    .parameters()
                    .iter()
                    .zip(inputs)
                    .filter_map(|(parameter, input)| {
                        (parameter.access == ParameterAccess::Mutable).then_some(*input)
                    })
                    .collect::<Vec<_>>();
                Self::push_access_specs(self, &mut specs, reads, writes);
                Self::push_owned_output_specs(self, &mut specs, outputs);
            }
            NodeKind::Copy | NodeKind::RepresentationConvert { .. } => {
                Self::push_access_specs(
                    self,
                    &mut specs,
                    inputs.first().copied(),
                    std::iter::empty(),
                );
                Self::push_owned_output_specs(self, &mut specs, outputs);
            }
            NodeKind::ElementWrite { authority } => {
                let place = *inputs
                    .first()
                    .unwrap_or_else(|| panic!("checked element write has no place"));
                Self::push_access_specs(self, &mut specs, [place], std::iter::empty());
                let representation = self
                    .tensor_representation(place)
                    .unwrap_or_else(|| panic!("checked element write place is not a tensor"));
                if let Some(authority) = authority {
                    assert_eq!(
                        &participants,
                        authority.participants(),
                        "checked write authority participant domain changed during monomorphization"
                    );
                }
                specs.push((
                    Some(place),
                    SemanticEventKind::Write(authority.clone()),
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
            NodeKind::Store { authority } => {
                let destination = *inputs
                    .first()
                    .unwrap_or_else(|| panic!("checked store has no destination"));
                let source = *inputs
                    .get(1)
                    .unwrap_or_else(|| panic!("checked store has no source"));
                Self::push_access_specs(
                    self,
                    &mut specs,
                    [destination, source],
                    std::iter::empty(),
                );
                let representation = self
                    .tensor_representation(destination)
                    .unwrap_or_else(|| panic!("checked store destination is not a tensor"));
                if let Some(authority) = authority {
                    assert_eq!(
                        &participants,
                        authority.participants(),
                        "checked store authority participant domain changed during monomorphization"
                    );
                }
                specs.push((
                    Some(destination),
                    SemanticEventKind::Write(authority.clone()),
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
        }
        if may_fail {
            specs.push((
                None,
                SemanticEventKind::MayFail,
                None,
                VisibilityScope::Participant,
                NumericalOutcome::Deterministic,
            ));
        }
        let mut prior = self
            .last_event
            .get(&node.region())
            .cloned()
            .unwrap_or_default();
        let mut events = Vec::with_capacity(specs.len());
        let location_indices = match kind {
            NodeKind::ElementRead => inputs[1..].to_vec(),
            NodeKind::ElementWrite { .. } => inputs[1..inputs.len() - 1].to_vec(),
            _ => Vec::new(),
        };
        for (ordinal, (place, access, representation, visibility, numerical)) in
            specs.into_iter().enumerate()
        {
            let id = SemanticEventId::new(
                node,
                u32::try_from(ordinal).expect("node has more than u32::MAX semantic events"),
            );
            let dependencies = prior.clone();
            events.push(SemanticEvent::checked(
                id,
                place,
                location_indices.clone(),
                access,
                representation,
                participants.clone(),
                dependencies,
                visibility,
                numerical,
            ));
            prior = vec![id];
        }
        if !events.is_empty() {
            self.last_event.insert(node.region(), prior);
        }
        events
    }

    fn push_access_specs(
        &mut self,
        specs: &mut Vec<(
            Option<SemanticValueId>,
            SemanticEventKind,
            Option<RepresentationId>,
            VisibilityScope,
            NumericalOutcome,
        )>,
        reads: impl IntoIterator<Item = SemanticValueId>,
        writes: impl IntoIterator<Item = SemanticValueId>,
    ) {
        for place in reads {
            if let Some(representation) = self.tensor_representation(place) {
                specs.push((
                    Some(place),
                    SemanticEventKind::Read,
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
        }
        for place in writes {
            let representation = self
                .tensor_representation(place)
                .unwrap_or_else(|| panic!("checked write place is not a tensor"));
            specs.push((
                Some(place),
                SemanticEventKind::Write(None),
                Some(representation),
                VisibilityScope::Participant,
                NumericalOutcome::Deterministic,
            ));
        }
    }

    fn push_owned_output_specs(
        &self,
        specs: &mut Vec<(
            Option<SemanticValueId>,
            SemanticEventKind,
            Option<RepresentationId>,
            VisibilityScope,
            NumericalOutcome,
        )>,
        outputs: &[SemanticValueId],
    ) {
        for output in outputs {
            if let Some(representation) = self.tensor_representation(*output) {
                specs.push((
                    Some(*output),
                    SemanticEventKind::Write(None),
                    Some(representation),
                    VisibilityScope::Participant,
                    NumericalOutcome::Deterministic,
                ));
            }
        }
    }

    fn semantic_type(&mut self, ty: &ValueType) -> SemanticType {
        match ty {
            ValueType::Scalar(dtype) => SemanticType::Scalar(*dtype),
            ValueType::Integer => SemanticType::Integer,
            ValueType::Index { bound } => {
                let bound = self.transfer_int(*bound);
                SemanticType::Index {
                    bound: self.builder.arena.nat_from_int(bound),
                }
            }
            ValueType::Range { bound } => {
                let bound = self.transfer_int(*bound);
                SemanticType::Range {
                    bound: self.builder.arena.nat_from_int(bound),
                }
            }
            ValueType::Tensor(tensor) => {
                let representation = self.resolve_elem(&tensor.elem);
                let axes = tensor
                    .axes
                    .iter()
                    .map(|axis| {
                        let axis = self.transfer_int(*axis);
                        self.builder.arena.nat_from_int(axis)
                    })
                    .collect();
                SemanticType::Tensor(TensorSemantics {
                    representation,
                    axes,
                    storage: TensorStorage::Computed,
                })
            }
            ValueType::Tuple(items) => {
                SemanticType::Tuple(items.iter().map(|item| self.semantic_type(item)).collect())
            }
            ValueType::Opaque { capability, name } => SemanticType::Opaque {
                capability: *capability,
                name,
            },
            ValueType::Void => SemanticType::Void,
        }
    }

    fn resolve_elem(&self, elem: &Elem) -> RepresentationId {
        match elem {
            Elem::Dtype(dtype) => crate::registry::dense(*dtype),
            Elem::Repr(representation) => *representation,
            Elem::Param(name) => *self
                .elements
                .get(name)
                .unwrap_or_else(|| panic!("unbound element parameter in monomorphization")),
        }
    }

    fn transfer_int(&mut self, expression: IntExpr) -> IntExpr {
        let symbols = &self.symbols;
        let mut map =
            |symbol: SymbolId, _: &mut ExprArena| {
                AnyExpr::Int(*symbols.get(&symbol).unwrap_or_else(|| {
                    panic!("checked expression contains an unbound body symbol")
                }))
            };
        let transferred = xfer::transfer_int(
            &self.definition.arena,
            expression,
            &mut self.builder.arena,
            &mut map,
        );
        for symbol in self.builder.arena.free_symbols(transferred.into()) {
            if let crate::expr::SymbolKind::RuntimeValue(value) = self.builder.arena.symbol_kind(symbol) {
                assert_eq!(value.function(), self.id,
                    "semantic expression captured a runtime value from another function");
            }
        }
        transferred
    }

    fn runtime_int(&mut self, value: SemanticValueId) -> IntExpr {
        if let Some(expression) = self.runtime_values.get(&value) {
            return *expression;
        }
        let (_, expression) = self.builder.arena.runtime_value(value);
        self.runtime_values.insert(value, expression);
        expression
    }

    /// Install an already evaluated value at a source-owned destination.
    /// Generic elements have been resolved here, so the operation either has
    /// its concrete source meaning or receives a source specialization error.
    fn install_value(
        &mut self,
        region: RegionId,
        value: SemanticValueId,
        expected: &SemanticType,
        span: crate::span::Span,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        let actual = self.values[value.index()].ty.clone();
        if &actual == expected {
            return Ok(value);
        }
        let module = self.module;
        let file = self.definition.file;
        let unsupported = || source_error(
            module, file, span,
            "this element binding has no defined source installation conversion",
        );
        match (expected, actual) {
            (SemanticType::Scalar(target), SemanticType::Scalar(source)) if source == *target => Ok(value),
            (SemanticType::Scalar(target), SemanticType::Scalar(source))
                if source.is_float() && target.is_float() => Ok(self.emit(
                    region,
                    NodeKind::Primitive(PrimitiveId::Cast(*target)),
                    vec![value],
                    vec![SemanticType::Scalar(*target)],
                    span,
                )[0]),
            (SemanticType::Tensor(target), SemanticType::Tensor(source)) => {
                if source.axes.len() != target.axes.len() { return Err(unsupported()); }
                if source.representation == target.representation {
                    return Ok(value);
                }
                let source_dtype = match crate::registry::representation_info(source.representation).kind {
                    crate::registry::RepresentationKind::Dense(dtype) => Some(dtype),
                    _ => None,
                };
                let target_dtype = match crate::registry::representation_info(target.representation).kind {
                    crate::registry::RepresentationKind::Dense(dtype) => Some(dtype),
                    _ => None,
                };
                match (source_dtype, target_dtype) {
                    (Some(from), Some(to)) if from.is_float() && to.is_float() => Ok(self.emit(
                        region,
                        NodeKind::Elementwise(PrimitiveId::Cast(to)),
                        vec![value],
                        vec![SemanticType::Tensor(TensorSemantics {
                            representation: target.representation,
                            axes: source.axes,
                            storage: TensorStorage::Computed,
                        })],
                        span,
                    )[0]),
                    _ => Err(unsupported()),
                }
            }
            (SemanticType::Tuple(target), SemanticType::Tuple(source)) if target.len() == source.len() => {
                let mut converted = Vec::with_capacity(target.len());
                for (index, (target, source)) in target.iter().zip(source).enumerate() {
                    let component = self.emit(region, NodeKind::TupleGet { index: index as u32 },
                        vec![value], vec![source], span)[0];
                    converted.push(self.install_value(region, component, target, span)?);
                }
                Ok(self.emit(region, NodeKind::TuplePack, converted,
                    vec![SemanticType::Tuple(target.clone())], span)[0])
            }
            (SemanticType::Index { .. }, SemanticType::Index { .. })
            | (SemanticType::Range { .. }, SemanticType::Range { .. })
            | (SemanticType::Integer, SemanticType::Integer)
            | (SemanticType::Void, SemanticType::Void) => Ok(value),
            _ => Err(unsupported()),
        }
    }

    fn promote_operand(
        &mut self,
        region: RegionId,
        value: SemanticValueId,
        dtype: DType,
        span: crate::span::Span,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        let actual = self.values[value.index()].ty.clone();
        match actual {
            SemanticType::Scalar(from) if from == dtype => Ok(value),
            SemanticType::Scalar(from) if from.is_float() && dtype.is_float() => Ok(self.emit(
                region, NodeKind::Primitive(PrimitiveId::Cast(dtype)), vec![value],
                vec![SemanticType::Scalar(dtype)], span,
            )[0]),
            SemanticType::Tensor(tensor) => {
                let target = crate::registry::dense(dtype);
                if tensor.representation == target { return Ok(value); }
                let info = crate::registry::representation_info(tensor.representation);
                let permitted = match info.kind {
                    crate::registry::RepresentationKind::Dense(from) => from.is_float() && dtype.is_float(),
                    crate::registry::RepresentationKind::Packed(_) => dtype == DType::F32
                        && crate::registry::decode_recipe(tensor.representation, dtype).is_some(),
                    crate::registry::RepresentationKind::External(_) => false,
                };
                if permitted {
                    Ok(self.emit(region, NodeKind::Elementwise(PrimitiveId::Cast(dtype)), vec![value],
                        vec![SemanticType::Tensor(TensorSemantics {
                            representation: target, axes: tensor.axes, storage: TensorStorage::Computed,
                        })], span)[0])
                } else {
                    Err(source_error(self.module, self.definition.file, span,
                        "this element binding has no defined arithmetic operand conversion"))
                }
            }
            _ => Err(source_error(self.module, self.definition.file, span,
                "this source arithmetic has no compatible operand conversion")),
        }
    }

    fn compound_value(
        &mut self,
        region: RegionId,
        old: SemanticValueId,
        rhs: SemanticValueId,
        op: AssignOp,
        span: crate::span::Span,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        let binary = match op {
            AssignOp::Add => BinaryOp::Add,
            AssignOp::Sub => BinaryOp::Sub,
            AssignOp::Mul => BinaryOp::Mul,
            AssignOp::Assign => unreachable!(),
        };
        let target = self.values[old.index()].ty.clone();
        let rhs_ty = self.values[rhs.index()].ty.clone();
        if let (SemanticType::Scalar(dtype @ (DType::I32 | DType::U32)),
            SemanticType::Integer | SemanticType::Index { .. }) = (&target, &rhs_ty) {
            let projected = self.emit(region, NodeKind::Primitive(PrimitiveId::Cast(*dtype)),
                vec![rhs], vec![SemanticType::Scalar(*dtype)], span)[0];
            return Ok(self.emit(region, NodeKind::Primitive(PrimitiveId::Binary(binary)),
                vec![old, projected], vec![target], span)[0]);
        }
        let dtype = |ty: &SemanticType| match ty {
            SemanticType::Scalar(dtype) => Some(*dtype),
            SemanticType::Tensor(tensor) => Some(crate::registry::representation_info(tensor.representation).decoded),
            _ => None,
        };
        let (Some(left), Some(right)) = (dtype(&target), dtype(&rhs_ty)) else {
            return Err(source_error(self.module, self.definition.file, span,
                "this source compound arithmetic has no compatible operands"));
        };
        let working = DType::promote(left, right).ok_or_else(|| source_error(
            self.module, self.definition.file, span,
            "this element binding has no defined compound arithmetic promotion",
        ))?;
        let left = self.promote_operand(region, old, working, span)?;
        let right = self.promote_operand(region, rhs, working, span)?;
        let result_type = match &target {
            SemanticType::Scalar(_) => SemanticType::Scalar(working),
            SemanticType::Tensor(tensor) => SemanticType::Tensor(TensorSemantics {
                representation: crate::registry::dense(working),
                axes: tensor.axes.clone(), storage: TensorStorage::Computed,
            }),
            _ => unreachable!("numeric compound destination has no scalar or tensor type"),
        };
        let result = self.emit(region,
            self.primitive_kind(&PrimitiveId::Binary(binary), &result_type),
            vec![left, right], vec![result_type], span)[0];
        self.install_value(region, result, &target, span)
    }

    fn lower_block(
        &mut self,
        region: RegionId,
        block: &ir::Block,
    ) -> Result<Vec<SemanticValueId>, SourceDiagnostic> {
        for statement in &block.statements {
            self.lower_stmt(region, statement)?;
        }
        Ok(match &block.terminator {
            ir::Terminator::Continue => Vec::new(),
            ir::Terminator::Return(values) => {
                let mut out = Vec::new();
                for value in values {
                    let value = self.lower_expr(region, value)?;
                    self.flatten_value(region, value, value, &mut out, block)?;
                }
                out
            }
        })
    }

    fn flatten_value(
        &mut self,
        region: RegionId,
        value: SemanticValueId,
        _root: SemanticValueId,
        output: &mut Vec<SemanticValueId>,
        block: &ir::Block,
    ) -> Result<(), SourceDiagnostic> {
        let ty = self.values[value.index()].ty.clone();
        match ty {
            SemanticType::Tuple(items) => {
                for (index, item) in items.into_iter().enumerate() {
                    let projected = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![item],
                        self.values[value.index()].span,
                    )[0];
                    self.flatten_value(region, projected, projected, output, block)?;
                }
            }
            SemanticType::Void => {}
            _ => output.push(value),
        }
        Ok(())
    }

    fn lower_stmt(
        &mut self,
        region: RegionId,
        statement: &ir::Stmt,
    ) -> Result<(), SourceDiagnostic> {
        match statement {
            ir::Stmt::Let { pattern, value, .. } => {
                let span = value.span;
                let value = self.lower_expr(region, value)?;
                self.bind_pattern(region, pattern, value, span)?;
            }
            ir::Stmt::Assign {
                place,
                op,
                value,
                value_symbols,
                authorities,
            } => {
                fn contains_element(place: &ir::Place, region: &ir::Place) -> bool {
                    match place {
                        ir::Place::Local(_) => false,
                        ir::Place::Element { .. } => place == region,
                        ir::Place::Tuple(places) => {
                            places.iter().any(|place| contains_element(place, region))
                        }
                    }
                }
                // The checker issues capabilities only for participants that
                // share the destination storage. A tensor allocated inside a
                // parallel iteration is private to that participant and needs
                // no authority for it, even when the write is nested in loops.
                // Preserve that checked ownership decision instead of inferring
                // sharing from this lowering's parallel nesting depth.
                for (index, authority) in authorities.iter().enumerate() {
                    assert!(
                        contains_element(place, authority.region()),
                        "checked write authority names a different destination"
                    );
                    assert!(
                        !authorities[..index]
                            .iter()
                            .any(|earlier| earlier.region() == authority.region()),
                        "checked write authority is duplicated"
                    );
                }
                let span = value.span;
                let value = self.lower_expr(region, value)?;
                self.assign(region, place, *op, value, authorities, span)?;
                for (local, symbol) in value_symbols {
                    let value = self.local(*local);
                    let actual = self.runtime_int(value);
                    self.symbols.insert(*symbol, actual);
                }
            }
            ir::Stmt::Evaluate(expression) => {
                self.lower_expr(region, expression)?;
            }
            ir::Stmt::If {
                condition,
                then_body,
                else_body,
                capture_symbols,
                join_symbols,
            } => self.lower_if(region, condition, then_body, else_body, capture_symbols, join_symbols)?,
            ir::Stmt::Loop {
                kind,
                binder,
                start,
                end,
                body,
                value_symbols,
                initialization,
            } => self.lower_loop(region, *kind, *binder, start, end, body, value_symbols, initialization)?,
        }
        Ok(())
    }

    fn bind_pattern(
        &mut self,
        region: RegionId,
        pattern: &ir::Pattern,
        value: SemanticValueId,
        span: crate::span::Span,
    ) -> Result<(), SourceDiagnostic> {
        match pattern {
            ir::Pattern::Local(local) => {
                self.locals[local.index()] = Some(value);
                if let Some(source_symbol) = self.definition.body.locals[local.index()].symbol {
                    let runtime = self.runtime_int(value);
                    self.symbols.insert(source_symbol, runtime);
                }
            }
            ir::Pattern::Tuple(patterns) => {
                let SemanticType::Tuple(items) = self.values[value.index()].ty.clone() else {
                    panic!("checked tuple pattern received a non-tuple value")
                };
                assert_eq!(
                    patterns.len(),
                    items.len(),
                    "checked tuple pattern arity changed during monomorphization"
                );
                for (index, (pattern, ty)) in patterns.iter().zip(items).enumerate() {
                    let component = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![ty],
                        span,
                    )[0];
                    self.bind_pattern(region, pattern, component, span)?;
                }
            }
        }
        Ok(())
    }

    fn local_place(
        &mut self,
        region: RegionId,
        place: &super::ownership::LocalPlace,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let mut value = self.local(place.local);
        for index in &place.path {
            let SemanticType::Tuple(parts) = &self.values[value.index()].ty else {
                panic!("checked local projection is not a tuple")
            };
            let ty = parts[*index].clone();
            value = self.emit(
                region,
                NodeKind::TupleGet {
                    index: *index as u32,
                },
                vec![value],
                vec![ty],
                span,
            )[0];
        }
        value
    }
    fn replace_local_place(
        &mut self,
        region: RegionId,
        place: &super::ownership::LocalPlace,
        value: SemanticValueId,
        span: crate::span::Span,
    ) {
        fn replace(
            builder: &mut FunctionLowering<'_, '_>,
            region: RegionId,
            old: SemanticValueId,
            path: &[usize],
            value: SemanticValueId,
            span: crate::span::Span,
        ) -> SemanticValueId {
            let Some((index, rest)) = path.split_first() else {
                return value;
            };
            let ty = builder.values[old.index()].ty.clone();
            let SemanticType::Tuple(parts) = &ty else {
                panic!("checked local replacement is not a tuple")
            };
            let mut items = Vec::with_capacity(parts.len());
            for (ordinal, part) in parts.iter().enumerate() {
                let current = builder.emit(
                    region,
                    NodeKind::TupleGet {
                        index: ordinal as u32,
                    },
                    vec![old],
                    vec![part.clone()],
                    span,
                )[0];
                items.push(if ordinal == *index {
                    replace(builder, region, current, rest, value, span)
                } else {
                    current
                });
            }
            builder.emit(region, NodeKind::TuplePack, items, vec![ty], span)[0]
        }
        let old = self.local(place.local);
        let updated = replace(self, region, old, &place.path, value, span);
        self.locals[place.local.index()] = Some(updated);
    }

    fn assignment_place_type(
        &self,
        place: &ir::Place,
    ) -> SemanticType {
        let local_type = |local: &super::ownership::LocalPlace| {
            let value = self.local(local.local);
            let mut ty = &self.values[value.index()].ty;
            for index in &local.path {
                let SemanticType::Tuple(parts) = ty else {
                    panic!("checked local projection is not a tuple")
                };
                ty = &parts[*index];
            }
            ty.clone()
        };
        match place {
            ir::Place::Local(local) => local_type(local),
            ir::Place::Element { root, indices } => {
                assert!(indices.is_empty(), "tuple assignment only installs whole named tensor state");
                local_type(root)
            }
            ir::Place::Tuple(places) => SemanticType::Tuple(places.iter()
                .map(|place| self.assignment_place_type(place)).collect()),
        }
    }

    fn assign(
        &mut self,
        region: RegionId,
        place: &ir::Place,
        op: AssignOp,
        value: SemanticValueId,
        authorities: &[ir::ExclusiveWriteCapability],
        span: crate::span::Span,
    ) -> Result<(), SourceDiagnostic> {
        match place {
            ir::Place::Local(local) => {
                let value = if op == AssignOp::Assign {
                    let target = self.local_place(region, local, span);
                    let expected = self.values[target.index()].ty.clone();
                    self.install_value(region, value, &expected, span)?
                } else {
                    let old = self.local_place(region, local, span);
                    self.compound_value(region, old, value, op, span)?
                };
                self.replace_local_place(region, local, value, span);
            }
            ir::Place::Tuple(places) => {
                assert_eq!(op, AssignOp::Assign, "checked tuple assignment only uses `=`");
                let target = self.assignment_place_type(place);
                // Convert the complete RHS before installing any tuple leaf.
                let value = self.install_value(region, value, &target, span)?;
                let SemanticType::Tuple(items) = self.values[value.index()].ty.clone() else {
                    panic!("checked tuple assignment received a non-tuple value")
                };
                assert_eq!(
                    places.len(),
                    items.len(),
                    "checked tuple assignment arity changed during monomorphization"
                );
                for (index, (place, ty)) in places.iter().zip(items).enumerate() {
                    let component = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![ty],
                        span,
                    )[0];
                    self.assign(region, place, op, component, authorities, span)?;
                }
            }
            ir::Place::Element { root, indices } => {
                let supplied =
                    authorities
                        .iter()
                        .find(|authority| authority.region() == place)
                        .map(|authority| {
                            authority
                                .participants()
                                .iter()
                                .map(|participant| {
                                    *self.participant_bindings.get(participant).unwrap_or_else(|| {
                                    panic!("checked write authority names an inactive participant")
                                })
                                })
                                .collect::<Vec<_>>()
                        });
                let base = self.local_place(region, root, span);
                let semantic_participants = supplied.map(|supplied| {
                    assert!(!supplied.is_empty());
                    assert!(supplied
                        .iter()
                        .all(|participant| self.parallel_participants.contains(participant)));
                    let participants = ParticipantDomain::Parallel(supplied.into_boxed_slice());
                    participants
                });
                if let SemanticType::Tensor(tensor) = &self.values[base.index()].ty {
                    if crate::registry::representation_info(tensor.representation).access
                        != crate::registry::RepresentationAccess::ReadWrite
                    {
                        return Err(SourceDiagnostic::new(
                            &self.module.sources.files()[self.definition.file],
                            span,
                            crate::checked::DiagnosticRule::Type,
                            format!(
                                "representation `{}` is decode-only and has no canonical write contract",
                                crate::registry::representation_info(tensor.representation).name
                            ),
                        ));
                    }
                }
                let mut inputs = vec![base];
                let mut slice_axes = Vec::with_capacity(indices.len());
                for (axis, index) in indices.iter().enumerate() {
                    match index {
                        ir::Index::Point {
                            value: point,
                            runtime_check,
                        } => {
                            let point = self.lower_expr(region, point)?;
                            if *runtime_check {
                                self.emit_point_check(region, base, axis, point, span);
                            }
                            inputs.push(point);
                            slice_axes.push(SliceAxis::Point {
                                value: ScalarRef::Value(point),
                                runtime_check: *runtime_check,
                            });
                        }
                        ir::Index::Range {
                            start,
                            end,
                            check_start,
                            check_order,
                            check_end,
                        } => {
                            let start_value = match start {
                                Some(start) => Some(self.lower_expr(region, start)?),
                                None => None,
                            };
                            let end_value = match end {
                                Some(end) => Some(self.lower_expr(region, end)?),
                                None => None,
                            };
                            self.emit_range_checks(
                                region,
                                base,
                                axis,
                                start_value,
                                end_value,
                                (*check_start, *check_order, *check_end),
                                span,
                            );
                            if let Some(start) = start_value {
                                inputs.push(start);
                            }
                            if let Some(end) = end_value {
                                inputs.push(end);
                            }
                            slice_axes.push(if start_value.is_none() && end_value.is_none() {
                                SliceAxis::Full
                            } else {
                                SliceAxis::Range {
                                    start: start_value.map(ScalarRef::Value),
                                    end: end_value.map(ScalarRef::Value),
                                    check_start: *check_start,
                                    check_order: *check_order,
                                    check_end: *check_end,
                                }
                            });
                        }
                    }
                }
                let stored_value = if op == AssignOp::Assign {
                    let SemanticType::Tensor(base_tensor) = &self.values[base.index()].ty else {
                        panic!("checked element place has a non-tensor root")
                    };
                    let expected = match &self.values[value.index()].ty {
                        SemanticType::Tensor(tensor) => SemanticType::Tensor(TensorSemantics {
                            representation: base_tensor.representation,
                            axes: tensor.axes.clone(),
                            storage: TensorStorage::Computed,
                        }),
                        SemanticType::Scalar(_) => SemanticType::Scalar(
                            crate::registry::representation_info(base_tensor.representation).decoded,
                        ),
                        _ => panic!("checked tensor store value is not numerical"),
                    };
                    self.install_value(region, value, &expected, span)?
                } else {
                    let representation = match &self.values[base.index()].ty {
                        SemanticType::Tensor(tensor) => tensor.representation,
                        _ => panic!("checked element place has a non-tensor root"),
                    };
                    let dtype = crate::registry::representation_info(representation).decoded;
                    let whole_tensor = indices.is_empty();
                    let old = if whole_tensor {
                        base
                    } else {
                        self.emit(
                            region,
                            NodeKind::ElementRead,
                            inputs.clone(),
                            vec![SemanticType::Scalar(dtype)],
                            span,
                        )[0]
                    };
                    self.compound_value(region, old, value, op, span)?
                };
                let tensor_store = matches!(
                    self.values[stored_value.index()].ty,
                    SemanticType::Tensor(_)
                );
                let (kind, store_inputs) = if tensor_store {
                    let base_tensor = match &self.values[base.index()].ty {
                        SemanticType::Tensor(tensor) => tensor.clone(),
                        _ => panic!("checked store base is not a tensor"),
                    };
                    slice_axes.extend(
                        (slice_axes.len()..base_tensor.axes.len()).map(|_| SliceAxis::Full),
                    );
                    let transform = ViewTransform::Slice { axes: slice_axes };
                    let stored_tensor = match &self.values[stored_value.index()].ty {
                        SemanticType::Tensor(tensor) => tensor,
                        _ => unreachable!(),
                    };
                    let destination = self.emit(
                        region,
                        NodeKind::View(transform.clone()),
                        inputs,
                        vec![SemanticType::Tensor(TensorSemantics {
                            representation: base_tensor.representation,
                            axes: stored_tensor.axes.clone(),
                            storage: TensorStorage::View { base, transform },
                        })],
                        span,
                    )[0];
                    (
                        NodeKind::Store {
                            authority: semantic_participants.map(|participants| {
                                ExclusiveWriteCapability::checked(
                                    destination,
                                    Vec::new(),
                                    participants,
                                )
                            }),
                        },
                        vec![destination, stored_value],
                    )
                } else {
                    let semantic_indices = inputs[1..].to_vec();
                    inputs.push(stored_value);
                    (
                        NodeKind::ElementWrite {
                            authority: semantic_participants.map(|participants| {
                                ExclusiveWriteCapability::checked(
                                    base,
                                    semantic_indices,
                                    participants,
                                )
                            }),
                        },
                        inputs,
                    )
                };
                let ty = self.values[base.index()].ty.clone();
                let updated = self.emit(region, kind, store_inputs, vec![ty], span)[0];
                self.replace_local_place(region, root, updated, span);
            }
        }
        Ok(())
    }

    fn local(&self, local: ir::LocalId) -> SemanticValueId {
        self.locals[local.index()]
            .unwrap_or_else(|| panic!("checked local used before semantic definition"))
    }

    fn primitive_kind(&self, primitive: &PrimitiveId, output: &SemanticType) -> NodeKind {
        match primitive {
            PrimitiveId::TuplePack => NodeKind::TuplePack,
            PrimitiveId::TupleGet(index) => NodeKind::TupleGet { index: *index },
            PrimitiveId::TensorAlloc => NodeKind::Alloc,
            PrimitiveId::Fill(value) => NodeKind::Fill { value: *value },
            PrimitiveId::Copy => NodeKind::Copy,
            PrimitiveId::RepresentationConvert(_) => {
                panic!("representation conversion is resolved during entry construction")
            }
            PrimitiveId::SliceView { .. } | PrimitiveId::Transpose | PrimitiveId::Reshape => {
                panic!("view primitives require their structural transform")
            }
            PrimitiveId::ElementRead { .. } => NodeKind::ElementRead,
            PrimitiveId::Atomic { .. } => {
                panic!("checked atomic primitive lacks its identity-bound authority")
            }
            PrimitiveId::Reduce {
                op,
                axis,
                unordered,
                ..
            } => NodeKind::Reduce {
                op: *op,
                axis: *axis,
                unordered: *unordered,
            },
            PrimitiveId::Extent { axis } => NodeKind::Extent { axis: *axis },
            PrimitiveId::Constant(_)
            | PrimitiveId::Symbolic(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select => {
                if matches!(output, SemanticType::Tensor(_)) {
                    NodeKind::Elementwise(primitive.clone())
                } else {
                    NodeKind::Primitive(primitive.clone())
                }
            }
        }
    }

    fn scalar_constant(
        &mut self,
        region: RegionId,
        value: i64,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Constant(ReferenceScalar::I32(
                i32::try_from(value).expect("source i32 constant exceeds its width"),
            ))),
            Vec::new(),
            vec![SemanticType::Scalar(DType::I32)],
            span,
        )[0]
    }

    fn extent_value(
        &mut self,
        region: RegionId,
        base: SemanticValueId,
        axis: usize,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Extent {
                axis: u32::try_from(axis).expect("tensor rank exceeds u32::MAX"),
            },
            vec![base],
            vec![SemanticType::Scalar(DType::I32)],
            span,
        )[0]
    }

    fn comparison(
        &mut self,
        region: RegionId,
        op: BinaryOp,
        left: SemanticValueId,
        right: SemanticValueId,
        span: crate::span::Span,
    ) -> SemanticValueId {
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Binary(op)),
            vec![left, right],
            vec![SemanticType::Scalar(DType::Bool)],
            span,
        )[0]
    }

    /// An operation check remains a source event even when its predicate is
    /// invocation-known: rejecting earlier would discard preceding effects.
    fn check(
        &mut self,
        region: RegionId,
        condition: SemanticValueId,
        reason: CheckReason,
        span: crate::span::Span,
    ) {
        self.emit(
            region,
            NodeKind::Check { reason },
            vec![condition],
            Vec::new(),
            span,
        );
    }

    /// The region parameter whose storage a tensor value aliases. This
    /// follows only identity-preserving view and mutation nodes; computed
    /// tensors never inherit an alias merely because they read a parameter.
    fn region_storage_origin(
        &self,
        region: RegionId,
        value: SemanticValueId,
        seen: &mut BTreeSet<SemanticValueId>,
    ) -> Option<usize> {
        if !seen.insert(value) {
            return None;
        }
        let work = self.regions[region.index()].as_ref()?;
        if let Some(ordinal) = work
            .parameters
            .iter()
            .position(|parameter| *parameter == value)
        {
            return Some(ordinal);
        }
        let ValueOrigin::Node(node) = self.values[value.index()].origin else {
            return None;
        };
        if node.region() != region {
            return None;
        }
        let aliased = match work.nodes[node.ordinal()].view() {
            SemanticNodeView::View { base, .. } => base,
            SemanticNodeView::ElementWrite { place, .. }
            | SemanticNodeView::Atomic { place, .. } => place,
            SemanticNodeView::Store { destination, .. } => destination,
            SemanticNodeView::Loop { carries, .. } => carries
                .iter()
                .find(|carry| carry.result == value)
                .map(|carry| carry.initial)?,
            _ => return None,
        };
        self.region_storage_origin(region, aliased, seen)
    }

    /// A mutation may replace the contents token while retaining the complete
    /// tensor descriptor. Sharing only a backing is insufficient: transpose,
    /// slice and reshape change the value transported across a control join.
    fn region_descriptor_origin(
        &self,
        region: RegionId,
        value: SemanticValueId,
        seen: &mut BTreeSet<SemanticValueId>,
    ) -> Option<usize> {
        if !seen.insert(value) {
            return None;
        }
        let work = self.regions[region.index()].as_ref()?;
        if let Some(ordinal) = work
            .parameters
            .iter()
            .position(|parameter| *parameter == value)
        {
            return Some(ordinal);
        }
        let ValueOrigin::Node(node) = self.values[value.index()].origin else {
            return None;
        };
        if node.region() != region {
            return None;
        }
        let unchanged = match work.nodes[node.ordinal()].view() {
            SemanticNodeView::View {
                base,
                transform: ViewTransform::Identity,
                ..
            } => base,
            SemanticNodeView::ElementWrite { place, .. }
            | SemanticNodeView::Atomic { place, .. } => place,
            SemanticNodeView::Store { destination, .. } => {
                match &self.values[destination.index()].ty {
                    SemanticType::Tensor(TensorSemantics {
                        storage: TensorStorage::View { base, .. },
                        ..
                    }) => *base,
                    _ => panic!("store result requires its explicit destination view"),
                }
            }
            _ => return None,
        };
        self.region_descriptor_origin(region, unchanged, seen)
    }

    fn emit_point_check(
        &mut self,
        region: RegionId,
        base: SemanticValueId,
        axis: usize,
        point: SemanticValueId,
        span: crate::span::Span,
    ) {
        let zero = self.scalar_constant(region, 0, span);
        let extent = self.extent_value(region, base, axis, span);
        let lower = self.comparison(region, BinaryOp::Ge, point, zero, span);
        self.check(region, lower, CheckReason::IndexBound, span);
        let upper = self.comparison(region, BinaryOp::Lt, point, extent, span);
        self.check(region, upper, CheckReason::IndexBound, span);
    }

    fn emit_range_checks(
        &mut self,
        region: RegionId,
        base: SemanticValueId,
        axis: usize,
        start: Option<SemanticValueId>,
        end: Option<SemanticValueId>,
        enabled: (bool, bool, bool),
        span: crate::span::Span,
    ) {
        if !enabled.0 && !enabled.1 && !enabled.2 {
            return;
        }
        let zero = self.scalar_constant(region, 0, span);
        let extent = self.extent_value(region, base, axis, span);
        let start = start.unwrap_or(zero);
        let end = end.unwrap_or(extent);
        if enabled.0 {
            let condition = self.comparison(region, BinaryOp::Ge, start, zero, span);
            self.check(region, condition, CheckReason::IndexBound, span);
        }
        if enabled.1 {
            let condition = self.comparison(region, BinaryOp::Le, start, end, span);
            self.check(region, condition, CheckReason::RangeOrder, span);
        }
        if enabled.2 {
            let condition = self.comparison(region, BinaryOp::Le, end, extent, span);
            self.check(region, condition, CheckReason::IndexBound, span);
        }
    }

    fn lower_expr(
        &mut self,
        region: RegionId,
        expression: &ir::Expr,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        match &expression.kind {
            ir::ExprKind::Local(local) => Ok(self.local(*local)),
            ir::ExprKind::Literal(literal) => {
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Primitive(PrimitiveId::Constant(*literal)),
                    Vec::new(),
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Dimension(ordinal) => {
                let symbolic = self.shapes
                    [usize::try_from(*ordinal).expect("dimension ordinal does not fit usize")];
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Primitive(PrimitiveId::Symbolic(symbolic)),
                    Vec::new(),
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::PlaneView { base, plane } => {
                let base = self.lower_expr(region, base)?;
                let mut ty = self.semantic_type(&expression.ty);
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: ViewTransform::Plane { plane: *plane },
                    };
                }
                Ok(self.emit(
                    region,
                    NodeKind::View(ViewTransform::Plane { plane: *plane }),
                    vec![base],
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Primitive { id, operands } => {
                self.lower_primitive(region, id, operands, expression)
            }
            ir::ExprKind::Atomic {
                op,
                place,
                indices,
                value,
                authority,
            } => {
                let ir::ExprKind::Local(binding) = &place.kind else {
                    panic!("checked atomic authority is not bound to a local place")
                };
                let ir::Place::Element {
                    root: expected_root,
                    indices: expected_indices,
                } = authority.region()
                else {
                    panic!("checked atomic authority is not bound to an element region")
                };
                let expected_points = expected_indices
                    .iter()
                    .map(|index| match index {
                        ir::Index::Point { value, .. } => value,
                        ir::Index::Range { .. } => {
                            panic!("checked atomic authority contains a range region")
                        }
                    })
                    .collect::<Vec<_>>();
                assert!(expected_points.into_iter().eq(indices.iter()));
                let actual = self.local(*binding);
                let expected = self.local_place(region, expected_root, expression.span);
                if self.region_storage_origin(region, actual, &mut BTreeSet::new())
                    != self.region_storage_origin(region, expected, &mut BTreeSet::new())
                    && actual != expected
                {
                    panic!("checked atomic authority is bound to different storage")
                }
                let checked_participants = authority
                    .participants()
                    .iter()
                    .map(|participant| {
                        *self
                            .participant_bindings
                            .get(participant)
                            .unwrap_or_else(|| {
                                panic!("checked atomic authority names an inactive participant")
                            })
                    })
                    .collect::<Vec<_>>();
                assert!(checked_participants
                    .iter()
                    .all(|participant| self.parallel_participants.contains(participant)));
                assert!(matches!(authority.order(), ir::CheckedAtomicOrder::Relaxed));
                match authority.scope() {
                    ir::CheckedAtomicScope::Participant => {
                        assert!(checked_participants.is_empty())
                    }
                    ir::CheckedAtomicScope::Participants(participants) => {
                        let scope =
                            participants
                                .iter()
                                .map(|participant| {
                                    *self.participant_bindings.get(participant).unwrap_or_else(|| {
                                    panic!("checked atomic scope names an inactive participant")
                                })
                                })
                                .collect::<Vec<_>>();
                        assert_eq!(scope, checked_participants);
                    }
                }
                assert!(matches!(
                    authority.publication(),
                    ir::CheckedAtomicPublication::CommandCompletion
                ));
                let representation = match &self.values[actual.index()].ty {
                    SemanticType::Tensor(tensor) => tensor.representation,
                    _ => panic!("checked atomic place is not a tensor"),
                };
                let dtype = crate::registry::representation_info(representation).decoded;
                let semantic_outcome = match authority.outcome() {
                    ir::CheckedAssociationOutcome::Exact => AssociationOutcome::Exact,
                    ir::CheckedAssociationOutcome::Reassociated { accumulator } => {
                        assert_eq!(accumulator, dtype);
                        AssociationOutcome::Reassociated { accumulator }
                    }
                };
                let expected_outcome = match op {
                    crate::intrinsics::AtomicOp::Add if dtype.is_float() => {
                        AssociationOutcome::Reassociated { accumulator: dtype }
                    }
                    crate::intrinsics::AtomicOp::Add
                    | crate::intrinsics::AtomicOp::Max
                    | crate::intrinsics::AtomicOp::Min => AssociationOutcome::Exact,
                };
                assert_eq!(semantic_outcome, expected_outcome);
                let participants = if checked_participants.is_empty() {
                    ParticipantDomain::Single
                } else {
                    ParticipantDomain::Parallel(checked_participants.into_boxed_slice())
                };
                let mut inputs = vec![actual];
                let mut semantic_indices = Vec::with_capacity(indices.len());
                for index in indices {
                    let index = self.lower_expr(region, index)?;
                    semantic_indices.push(index);
                    inputs.push(index);
                }
                let capability = AtomicCapability::checked(
                    actual,
                    semantic_indices,
                    participants,
                    semantic_outcome,
                );
                inputs.push(self.lower_expr(region, value)?);
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Atomic {
                        op: *op,
                        capability,
                    },
                    inputs,
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Intrinsic { id, args } => {
                let signature = crate::registry::intrinsic_signature(*id);
                match signature.execution {
                    crate::registry::IntrinsicExecution::WithinEnclosingParallel
                        if self.parallel_depth == 0 =>
                    {
                        return Err(SourceDiagnostic::new(
                            &self.module.sources.files()[self.definition.file],
                            expression.span,
                            crate::checked::DiagnosticRule::Type,
                            format!(
                                "intrinsic `{}` requires an enclosing parallel loop",
                                signature.name
                            ),
                        ));
                    }
                    crate::registry::IntrinsicExecution::WholeTensor { .. }
                        if self.parallel_depth != 0 =>
                    {
                        return Err(SourceDiagnostic::new(
                            &self.module.sources.files()[self.definition.file],
                            expression.span,
                            crate::checked::DiagnosticRule::Type,
                            format!(
                                "whole-tensor intrinsic `{}` cannot be nested in a parallel loop",
                                signature.name
                            ),
                        ));
                    }
                    crate::registry::IntrinsicExecution::WithinEnclosingParallel
                    | crate::registry::IntrinsicExecution::WholeTensor { .. } => {}
                }
                let mut inputs = Vec::new();
                for argument in args {
                    inputs.push(self.lower_expr(region, argument)?);
                }
                let ty = self.semantic_type(&expression.ty);
                Ok(self.emit(
                    region,
                    NodeKind::Intrinsic(*id),
                    inputs,
                    vec![ty],
                    expression.span,
                )[0])
            }
            ir::ExprKind::Call { call, args } => self.lower_call(region, call, args, expression),
        }
    }

    fn lower_primitive(
        &mut self,
        region: RegionId,
        primitive: &PrimitiveId,
        operands: &[ir::Expr],
        expression: &ir::Expr,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        let mut inputs = Vec::new();
        for operand in operands {
            inputs.push(self.lower_expr(region, operand)?);
        }
        let transferred;
        let primitive = if let PrimitiveId::Symbolic(expression) = primitive {
            transferred = PrimitiveId::Symbolic(self.transfer_int(*expression));
            &transferred
        } else {
            primitive
        };
        let mut ty = if let PrimitiveId::SliceView { indices } = primitive {
            let base = inputs[0];
            let SemanticType::Tensor(base_tensor) = self.values[base.index()].ty.clone() else {
                panic!("checked slice base is not a tensor")
            };
            let representation = match &expression.ty {
                ValueType::Tensor(tensor) => self.resolve_elem(&tensor.elem),
                _ => panic!("checked slice result is not a tensor"),
            };
            let mut operand = 1usize;
            let mut axes = Vec::new();
            for (axis, index) in indices.iter().enumerate() {
                let (start, end) = match *index {
                    crate::intrinsics::IndexSlot::Point { .. } => {
                        operand += 1;
                        continue;
                    }
                    crate::intrinsics::IndexSlot::Range { start, end, .. } => (start, end),
                    // The whole axis: a range with no operands.
                    crate::intrinsics::IndexSlot::Full => (false, false),
                };
                let start = if start {
                    let value = self.runtime_int(inputs[operand]);
                    operand += 1;
                    value
                } else {
                    self.builder.arena.int(0)
                };
                let end = if end {
                    let value = self.runtime_int(inputs[operand]);
                    operand += 1;
                    value
                } else {
                    self.builder.arena.int_from_nat(base_tensor.axes[axis])
                };
                let width = self.builder.arena.int_sub(end, start);
                axes.push(self.builder.arena.nat_from_int(width));
            }
            axes.extend(base_tensor.axes[indices.len()..].iter().copied());
            if let ValueType::Tensor(source_tensor) = &expression.ty {
                for (source_axis, semantic_axis) in source_tensor.axes.iter().zip(&axes) {
                    if let NodeView::Symbol(symbol) =
                        self.definition.arena.view(AnyExpr::Int(*source_axis))
                    {
                        if !self.symbols.contains_key(&symbol) {
                            let semantic_axis = self.builder.arena.int_from_nat(*semantic_axis);
                            self.symbols.insert(symbol, semantic_axis);
                        }
                    }
                }
            }
            SemanticType::Tensor(TensorSemantics {
                representation,
                axes,
                storage: TensorStorage::Computed,
            })
        } else if matches!(primitive, PrimitiveId::Fill(_)) {
            let ValueType::Tensor(source_tensor) = &expression.ty else {
                panic!("checked fill has a non-tensor result")
            };
            let SemanticType::Tensor(like) = &self.values[inputs[0].index()].ty else {
                panic!("checked fill shape source is not a tensor")
            };
            SemanticType::Tensor(TensorSemantics {
                representation: self.resolve_elem(&source_tensor.elem),
                axes: like.axes.clone(),
                storage: TensorStorage::Computed,
            })
        } else if matches!(primitive, PrimitiveId::TensorAlloc | PrimitiveId::Reshape) {
            // Tensor construction and reshape get their logical axes from the
            // already evaluated source operands. The checked type supplies
            // element/representation meaning; its formulas are proofs, not
            // a second way to compute these extent values.
            let ValueType::Tensor(source_tensor) = &expression.ty else {
                panic!("checked tensor constructor has a non-tensor result")
            };
            let representation = self.resolve_elem(&source_tensor.elem);
            let extents = if matches!(primitive, PrimitiveId::Reshape) {
                &inputs[1..]
            } else {
                inputs.as_slice()
            };
            let axes = extents
                .iter()
                .map(|value| {
                    let value = self.runtime_int(*value);
                    self.builder.arena.nat_from_int(value)
                })
                .collect();
            SemanticType::Tensor(TensorSemantics {
                representation,
                axes,
                storage: TensorStorage::Computed,
            })
        } else {
            self.semantic_type(&expression.ty)
        };
        if let PrimitiveId::Cast(dtype) = primitive {
            if let Some(input) = inputs.first() {
                if let SemanticType::Tensor(source) = &self.values[input.index()].ty {
                    let source_info = crate::registry::representation_info(source.representation);
                    let permitted = match &source_info.kind {
                        crate::registry::RepresentationKind::Dense(from) => from.is_numeric() && dtype.is_numeric(),
                        crate::registry::RepresentationKind::Packed(_) => *dtype == DType::F32
                            && crate::registry::decode_recipe(source.representation, *dtype).is_some(),
                        crate::registry::RepresentationKind::External(_) => false,
                    };
                    if !permitted {
                        return Err(source_error(self.module, self.definition.file, expression.span,
                            format!("cannot convert `{}` to `{}` elements; packed values decode with `f32(v)`",
                                source_info.name, dtype.name())));
                    }
                }
            }
        }
        if matches!(primitive, PrimitiveId::TensorAlloc) {
            if let SemanticType::Tensor(tensor) = &ty {
                if crate::registry::representation_info(tensor.representation).access
                    != crate::registry::RepresentationAccess::ReadWrite
                {
                    return Err(SourceDiagnostic::new(
                        &self.module.sources.files()[self.definition.file],
                        expression.span,
                        crate::checked::DiagnosticRule::Type,
                        format!(
                            "representation `{}` is decode-only and cannot be allocated as writable logical storage",
                            crate::registry::representation_info(tensor.representation).name
                        ),
                    ));
                }
            }
        }
        if matches!(primitive, PrimitiveId::RepresentationConvert(_)) {
            let [input] = inputs.as_slice() else {
                panic!("checked representation conversion has wrong arity")
            };
            let SemanticType::Tensor(source) = &self.values[input.index()].ty else {
                panic!("checked representation conversion source is not a tensor")
            };
            let SemanticType::Tensor(destination) = &mut ty else {
                panic!("checked representation conversion result is not a tensor")
            };
            let Some(conversion) = crate::registry::representation_conversion(
                source.representation,
                destination.representation,
            ) else {
                return Err(SourceDiagnostic::new(
                    &self.module.sources.files()[self.definition.file],
                    expression.span,
                    crate::checked::DiagnosticRule::Type,
                    format!(
                        "no exact representation conversion is registered from `{}` to `{}`",
                        crate::registry::representation_info(source.representation).name,
                        crate::registry::representation_info(destination.representation).name,
                    ),
                ));
            };
            destination.storage = TensorStorage::Owned;
            return Ok(self.emit(
                region,
                NodeKind::RepresentationConvert {
                    conversion: conversion.id,
                },
                vec![*input],
                vec![ty],
                expression.span,
            )[0]);
        }
        match primitive {
            PrimitiveId::ElementRead { checks } => {
                let base = inputs[0];
                for axis in 0..checks.len() {
                    self.emit_point_check(region, base, axis, inputs[axis + 1], expression.span);
                }
            }
            PrimitiveId::SliceView { indices } => {
                let base = inputs[0];
                let mut operand = 1;
                for (axis, slot) in indices.iter().enumerate() {
                    match slot {
                        crate::intrinsics::IndexSlot::Point { .. } => {
                            self.emit_point_check(
                                region,
                                base,
                                axis,
                                inputs[operand],
                                expression.span,
                            );
                            operand += 1;
                        }
                        // The whole axis: no operands and no checks.
                        crate::intrinsics::IndexSlot::Full => {}
                        crate::intrinsics::IndexSlot::Range { start, end, .. } => {
                            let start_value = if *start {
                                let value = inputs[operand];
                                operand += 1;
                                Some(value)
                            } else {
                                None
                            };
                            let end_value = if *end {
                                let value = inputs[operand];
                                operand += 1;
                                Some(value)
                            } else {
                                None
                            };
                            self.emit_range_checks(
                                region,
                                base,
                                axis,
                                start_value,
                                end_value,
                                (*start, *start || *end, *end),
                                expression.span,
                            );
                        }
                    }
                }
            }
            PrimitiveId::Reduce {
                op:
                    crate::intrinsics::ReduceOp::Max
                    | crate::intrinsics::ReduceOp::Min
                    | crate::intrinsics::ReduceOp::Argmax,
                axis,
                ..
            } => {
                let extent = self.extent_value(region, inputs[0], *axis as usize, expression.span);
                let zero = self.scalar_constant(region, 0, expression.span);
                let nonempty = self.comparison(region, BinaryOp::Gt, extent, zero, expression.span);
                self.check(
                    region,
                    nonempty,
                    CheckReason::Custom("reduction axis must be nonempty".to_owned()),
                    expression.span,
                );
            }
            PrimitiveId::Constant(_)
            | PrimitiveId::Symbolic(_)
            | PrimitiveId::TuplePack
            | PrimitiveId::TupleGet(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select
            | PrimitiveId::TensorAlloc
            | PrimitiveId::Fill(_)
            | PrimitiveId::Copy
            | PrimitiveId::RepresentationConvert(_)
            | PrimitiveId::Transpose
            | PrimitiveId::Reshape
            | PrimitiveId::Extent { .. }
            | PrimitiveId::Atomic { .. }
            | PrimitiveId::Reduce {
                op: crate::intrinsics::ReduceOp::Sum,
                ..
            } => {}
        }
        let kind = match primitive {
            PrimitiveId::SliceView { indices } => {
                let base = inputs[0];
                let mut scalars = inputs[1..].iter().copied();
                let mut axes = Vec::with_capacity(indices.len());
                for index in indices {
                    match index {
                        crate::intrinsics::IndexSlot::Point { .. } => axes.push(SliceAxis::Point {
                            value: ScalarRef::Value(
                                scalars.next().expect("checked slice omitted point operand"),
                            ),
                            runtime_check: true,
                        }),
                        crate::intrinsics::IndexSlot::Full => axes.push(SliceAxis::Full),
                        crate::intrinsics::IndexSlot::Range { start, end, .. } => {
                            let start = start.then(|| {
                                ScalarRef::Value(
                                    scalars.next().expect("checked slice omitted range start"),
                                )
                            });
                            let end = end.then(|| {
                                ScalarRef::Value(
                                    scalars.next().expect("checked slice omitted range end"),
                                )
                            });
                            if start.is_none() && end.is_none() {
                                axes.push(SliceAxis::Full);
                            } else {
                                axes.push(SliceAxis::Range {
                                    check_start: start.is_some(),
                                    check_order: start.is_some() || end.is_some(),
                                    check_end: end.is_some(),
                                    start,
                                    end,
                                });
                            }
                        }
                    }
                }
                let transform = ViewTransform::Slice { axes };
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: transform.clone(),
                    };
                }
                NodeKind::View(transform)
            }
            PrimitiveId::Transpose => {
                let base = inputs[0];
                let rank = match &self.values[base.index()].ty {
                    SemanticType::Tensor(t) => t.axes.len(),
                    _ => 0,
                };
                let transform = ViewTransform::Transpose {
                    permutation: (0..rank).rev().map(|i| i as u32).collect(),
                };
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: transform.clone(),
                    };
                }
                NodeKind::View(transform)
            }
            PrimitiveId::Reshape => {
                let base = inputs[0];
                let axes = match &ty {
                    SemanticType::Tensor(t) => t.axes.clone(),
                    _ => Vec::new(),
                };
                let transform = ViewTransform::Reshape { axes };
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::View {
                        base,
                        transform: transform.clone(),
                    };
                }
                NodeKind::View(transform)
            }
            PrimitiveId::TensorAlloc
            | PrimitiveId::Fill(_)
            | PrimitiveId::Copy
            | PrimitiveId::RepresentationConvert(_) => {
                if let SemanticType::Tensor(tensor) = &mut ty {
                    tensor.storage = TensorStorage::Owned;
                }
                self.primitive_kind(primitive, &ty)
            }
            PrimitiveId::ElementRead { .. }
            | PrimitiveId::Reduce { .. } => self.primitive_kind(primitive, &ty),
            PrimitiveId::Atomic { .. } => {
                panic!("checked atomic primitive bypassed its authority-bearing node")
            }
            PrimitiveId::Constant(_)
            | PrimitiveId::Symbolic(_)
            | PrimitiveId::TuplePack
            | PrimitiveId::TupleGet(_)
            | PrimitiveId::RangeMake
            | PrimitiveId::RangeStart
            | PrimitiveId::RangeEnd
            | PrimitiveId::Unary(_)
            | PrimitiveId::Binary(_)
            | PrimitiveId::Cast(_)
            | PrimitiveId::Math(_)
            | PrimitiveId::Select
            | PrimitiveId::Extent { .. } => self.primitive_kind(primitive, &ty),
        };
        // The slice's selected source coordinates are retained in its typed
        // transform. Other constructors keep their evaluated shape operands
        // as ordinary node dependencies.
        let node_inputs = if matches!(primitive, PrimitiveId::SliceView { .. }) {
            vec![inputs[0]]
        } else {
            inputs
        };
        let value = self.emit(region, kind, node_inputs, vec![ty], expression.span)[0];
        if matches!(primitive, PrimitiveId::RangeStart | PrimitiveId::RangeEnd | PrimitiveId::ElementRead { .. }) {
            // A checked integer projection denotes this one reached value.
            // Bounds and guards are proof facts; neither the tensor read nor
            // its source expression is reconstructed later.
            if let Some(expression) = expression.sym {
                let NodeView::Symbol(symbol) = self.definition.arena.view(expression.into()) else {
                    unreachable!("checked integer projection must own its symbolic value")
                };
                let runtime = self.runtime_int(value);
                self.symbols.insert(symbol, runtime);
            }
        }
        Ok(value)
    }

    fn lower_call(
        &mut self,
        region: RegionId,
        call: &ir::Call,
        args: &[ir::Expr],
        expression: &ir::Expr,
    ) -> Result<SemanticValueId, SourceDiagnostic> {
        // Explicit shape bindings precede ordinary arguments in source order.
        // A scalar binding's exact quantity is projected from the one value
        // just evaluated; its authored arithmetic is never replayed.
        let mut explicit_shapes = Vec::with_capacity(call.explicit_shapes.len());
        for (_, binding) in &call.explicit_shapes {
            let value = self.lower_expr(region, binding)?;
            let value = if matches!(self.values[value.index()].ty, SemanticType::Integer) {
                value
            } else {
                let exact = self.runtime_int(value);
                self.emit(region, NodeKind::Primitive(PrimitiveId::Symbolic(exact)),
                    Vec::new(), vec![SemanticType::Integer], binding.span)[0]
            };
            explicit_shapes.push(value);
        }
        let mut evaluated = Vec::with_capacity(args.len());
        for argument in args {
            evaluated.push(self.lower_expr(region, argument)?);
        }
        let contract = self.module.families[call.family.index()].contract;
        let checked_reference = call.candidates.iter()
            .find(|candidate| candidate.definition == contract)
            .expect("checked call has no portable reference binding");
        let order = &checked_reference.arg_order;
        let definition = &self.module.definitions[contract.index()];
        let mut captured_shapes = vec![None; definition.dimensions.len()];
        for (ordinal, plan) in &checked_reference.shape_plan {
            let ordinal = *ordinal as usize;
            let value = match plan {
                ir::ShapeBindingPlan::Explicit { binding } => explicit_shapes[*binding],
                ir::ShapeBindingPlan::Inferred { observation, formal_axis, coefficient, prior_dimensions } => {
                    let ir::ShapeObservation::TensorAxis { argument, path, axis, .. } = observation;
                    let mut tensor = evaluated[*argument];
                    let span = args[*argument].span;
                    for &index in path {
                        let SemanticType::Tuple(parts) = &self.values[tensor.index()].ty else {
                            panic!("checked shape observation lost its tuple path")
                        };
                        let ty = parts[index as usize].clone();
                        tensor = self.emit(region, NodeKind::TupleGet { index }, vec![tensor], vec![ty], span)[0];
                    }
                    let observed_value = self.emit(region, NodeKind::Extent { axis: *axis }, vec![tensor],
                        vec![SemanticType::Integer], span)[0];
                    let observed = self.runtime_int(observed_value);
                    let mut known = HashMap::new();
                    for &index in prior_dimensions {
                        let dimension = &definition.dimensions[index as usize];
                        let value = captured_shapes[index as usize]
                            .expect("checked shape inverse dependency was not captured first");
                        known.insert(dimension.symbol, self.runtime_int(value));
                    }
                    let unknown = definition.dimensions[ordinal].symbol;
                    let zero = self.builder.arena.int(0);
                    let mut map = |symbol: SymbolId, _: &mut ExprArena| {
                        AnyExpr::Int(if symbol == unknown { zero } else {
                            *known.get(&symbol).expect("checked shape inverse depends on uncaptured dimension")
                        })
                    };
                    let offset = xfer::transfer_int(&definition.arena, *formal_axis,
                        &mut self.builder.arena, &mut map);
                    if *coefficient == 1 && matches!(self.builder.arena.view(offset.into()), NodeView::IntConst(0)) {
                        captured_shapes[ordinal] = Some(observed_value);
                        continue;
                    }
                    let difference = self.builder.arena.int_sub(observed, offset);
                    let inverse = if *coefficient == 1 { difference } else {
                        let divisor = self.builder.arena.int(*coefficient);
                        self.builder.arena.int_div(difference, divisor)
                    };
                    self.emit(region, NodeKind::Primitive(PrimitiveId::Symbolic(inverse)),
                        Vec::new(), vec![SemanticType::Integer], span)[0]
                }
            };
            captured_shapes[ordinal] = Some(value);
        }
        let captured_shapes = captured_shapes.into_iter()
            .map(|value| value.expect("checked call omitted a dimension binding"))
            .collect::<Vec<_>>();
        let mut specs = Vec::new();
        for candidate in &call.candidates {
            if !candidate
                .requires_elems
                .iter()
                .all(|(caller_parameter, required)| {
                    self.elements.get(caller_parameter).copied()
                        == Some(self.resolve_elem(required))
                })
            {
                continue;
            }
            let shape_args = candidate
                .shape_args
                .iter()
                .map(|shape| self.transfer_int(*shape))
                .collect();
            let mut elements = self.elements.clone();
            for (name, elem) in &candidate.elem_args {
                elements.insert(name.clone(), self.resolve_elem(elem));
            }
            specs.push(CandidateSpec {
                definition: candidate.definition,
                shape_args,
                elements,
                call_site_proved: candidate.applicability_proven,
            });
            // Every candidate in a checked family has the same canonical
            // parameter order. The first candidate supplies the call's
            // argument permutation below.
        }
        let family = instantiate_family(self.builder, self.module, call.family.index(), specs, true)?;
        let reference = self.builder.families[family.index()]
            .as_ref().expect("instantiated call family")
            .candidates().iter()
            .find(|candidate| candidate.numerical == NumericalRole::Reference)
            .expect("instantiated call family has a reference")
            .function;
        let formal_leaves = self.builder.functions[reference.index()]
            .as_ref().expect("instantiated reference function")
            .parameters().iter().map(|parameter| {
                (parameter.origin.clone(),
                    self.builder.functions[reference.index()].as_ref().unwrap().value(parameter.value).ty.clone())
            }).collect::<Vec<_>>();
        let mut inputs = Vec::new();
        for (origin, expected) in formal_leaves {
            match origin {
                FunctionParameterOrigin::ShapeDimension { ordinal } => {
                    inputs.push(captured_shapes[ordinal as usize]);
                }
                FunctionParameterOrigin::Source { ordinal, path } => {
                    let source = ordinal as usize;
                    let mut value = evaluated[order[source]];
                    let argument_span = args[order[source]].span;
                    for index in path {
                        let SemanticType::Tuple(parts) = &self.values[value.index()].ty else {
                            panic!("checked call argument lost its tuple structure")
                        };
                        let ty = parts[index as usize].clone();
                        value = self.emit(region, NodeKind::TupleGet { index }, vec![value], vec![ty], argument_span)[0];
                    }
                    inputs.push(self.install_value(region, value, &expected, argument_span)?);
                }
            }
        }
        let ty = self.semantic_type(&expression.ty);
        let mut leaf_types = Vec::new();
        Self::flatten_call_type(&ty, &mut leaf_types);
        let leaves = self.emit(
            region,
            NodeKind::Call { family },
            inputs,
            leaf_types,
            expression.span,
        );
        let mut leaves = leaves.into_iter();
        let value = self.reconstruct_call_value(region, &ty, &mut leaves, expression.span);
        assert!(
            leaves.next().is_none(),
            "checked call result reconstruction left an unbound contract leaf"
        );
        Ok(value)
    }

    fn flatten_call_argument(
        &mut self,
        region: RegionId,
        value: SemanticValueId,
        ty: &SemanticType,
        leaves: &mut Vec<SemanticValueId>,
        span: crate::span::Span,
    ) {
        match ty {
            SemanticType::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    let component = self.emit(
                        region,
                        NodeKind::TupleGet {
                            index: u32::try_from(index)
                                .expect("tuple has more than u32::MAX items"),
                        },
                        vec![value],
                        vec![item.clone()],
                        span,
                    )[0];
                    self.flatten_call_argument(region, component, item, leaves, span);
                }
            }
            SemanticType::Void => {}
            _ => leaves.push(value),
        }
    }

    /// Function contracts and Call nodes carry results in canonical leaf
    /// order. Tuples are source structure, reconstructed explicitly after the
    /// call so implementation splicing never has to rediscover a result tree.
    fn flatten_call_type(ty: &SemanticType, leaves: &mut Vec<SemanticType>) {
        match ty {
            SemanticType::Tuple(items) => {
                for item in items {
                    Self::flatten_call_type(item, leaves);
                }
            }
            SemanticType::Void => {}
            leaf => leaves.push(leaf.clone()),
        }
    }

    fn reconstruct_call_value(
        &mut self,
        region: RegionId,
        ty: &SemanticType,
        leaves: &mut impl Iterator<Item = SemanticValueId>,
        span: crate::span::Span,
    ) -> SemanticValueId {
        match ty {
            SemanticType::Tuple(items) => {
                let inputs = items
                    .iter()
                    .map(|item| self.reconstruct_call_value(region, item, leaves, span))
                    .collect();
                self.emit(region, NodeKind::TuplePack, inputs, vec![ty.clone()], span)[0]
            }
            SemanticType::Void => self.emit(
                region,
                NodeKind::TuplePack,
                Vec::new(),
                vec![SemanticType::Tuple(Vec::new())],
                span,
            )[0],
            _ => leaves
                .next()
                .unwrap_or_else(|| panic!("checked call contract omitted a result leaf")),
        }
    }

    fn region_parameter_tree(
        &mut self,
        region: RegionId,
        ty: &SemanticType,
        span: crate::span::Span,
        leaves: &mut Vec<SemanticValueId>,
    ) -> SemanticValueId {
        match ty {
            SemanticType::Tuple(items) => {
                let inputs = items
                    .iter()
                    .map(|item| self.region_parameter_tree(region, item, span, leaves))
                    .collect();
                self.emit(region, NodeKind::TuplePack, inputs, vec![ty.clone()], span)[0]
            }
            SemanticType::Void => self.emit(
                region,
                NodeKind::TuplePack,
                Vec::new(),
                vec![SemanticType::Tuple(Vec::new())],
                span,
            )[0],
            leaf => {
                let value = self.region_parameter(region, leaf.clone(), span);
                leaves.push(value);
                value
            }
        }
    }

    fn lower_if(
        &mut self,
        parent_region: RegionId,
        condition: &ir::Expr,
        then_body: &ir::Block,
        else_body: &ir::Block,
        capture_symbols: &[(ir::LocalId, SymbolId)],
        join_symbols: &[(ir::LocalId, SymbolId)],
    ) -> Result<(), SourceDiagnostic> {
        let condition_value = self.lower_expr(parent_region, condition)?;
        let before = self.locals.clone();

        let then_region = self.reserve_region(RegionKind::Then);
        let else_region = self.reserve_region(RegionKind::Else);
        let predecessor = self
            .last_event
            .get(&parent_region)
            .cloned()
            .unwrap_or_default();
        self.last_event.insert(then_region, predecessor.clone());
        self.last_event.insert(else_region, predecessor);
        let mut captures = Vec::new();
        let mut then_locals = before.clone();
        let mut else_locals = before.clone();
        for (ordinal, value) in before.iter().copied().enumerate() {
            let Some(value) = value else { continue };
            let ty = self.values[value.index()].ty.clone();
            let span = self.definition.body.locals[ordinal].span;
            let mut parent_leaves = Vec::new();
            self.flatten_call_argument(parent_region, value, &ty, &mut parent_leaves, span);
            let mut then_leaves = Vec::new();
            let then_parameter =
                self.region_parameter_tree(then_region, &ty, span, &mut then_leaves);
            let mut else_leaves = Vec::new();
            let else_parameter =
                self.region_parameter_tree(else_region, &ty, span, &mut else_leaves);
            assert_eq!(parent_leaves.len(), then_leaves.len());
            assert_eq!(parent_leaves.len(), else_leaves.len());
            then_locals[ordinal] = Some(then_parameter);
            else_locals[ordinal] = Some(else_parameter);
            captures.push((
                ordinal,
                value,
                ty,
                parent_leaves,
                then_parameter,
                else_parameter,
            ));
        }
        let parent_formals = self.formals.clone();
        let mut captured_values = captures
            .iter()
            .flat_map(|(_, _, _, leaves, _, _)| leaves.iter().copied())
            .collect::<Vec<_>>();
        let mut then_formals = Vec::with_capacity(parent_formals.len());
        let mut else_formals = Vec::with_capacity(parent_formals.len());
        for value in &parent_formals {
            let ordinal = if let Some(ordinal) =
                captured_values.iter().position(|capture| capture == value)
            {
                ordinal
            } else {
                let ty = self.values[value.index()].ty.clone();
                let span = self.values[value.index()].span;
                self.region_parameter(then_region, ty.clone(), span);
                self.region_parameter(else_region, ty, span);
                captured_values.push(*value);
                captured_values.len() - 1
            };
            then_formals.push(self.region_mut(then_region).parameters[ordinal]);
            else_formals.push(self.region_mut(else_region).parameters[ordinal]);
        }
        let parent_symbols = self.symbols.clone();
        for (local, symbol) in capture_symbols {
            let parameter = then_locals[local.index()]
                .expect("checked branch capture has no then parameter");
            let actual = self.runtime_int(parameter);
            self.symbols.insert(*symbol, actual);
        }
        self.formals = then_formals;
        self.locals = then_locals;
        let then_result = self.lower_block(then_region, then_body)?;
        assert!(
            then_result.is_empty(),
            "checked branch contains an early return"
        );
        let then_after = self.locals.clone();

        self.symbols = parent_symbols.clone();
        for (local, symbol) in capture_symbols {
            let parameter = else_locals[local.index()]
                .expect("checked branch capture has no else parameter");
            let actual = self.runtime_int(parameter);
            self.symbols.insert(*symbol, actual);
        }
        self.formals = else_formals;
        self.locals = else_locals;
        let else_result = self.lower_block(else_region, else_body)?;
        assert!(
            else_result.is_empty(),
            "checked branch contains an early return"
        );
        let else_after = self.locals.clone();
        self.symbols = parent_symbols;
        self.formals = parent_formals;

        let mut changed = Vec::new();
        for (ordinal, _, ty, _, then_parameter, else_parameter) in &captures {
            let then_value = then_after[*ordinal];
            let else_value = else_after[*ordinal];
            if then_value != Some(*then_parameter) || else_value != Some(*else_parameter) {
                let then_value =
                    then_value.unwrap_or_else(|| panic!("checked branch loses a live local"));
                let else_value =
                    else_value.unwrap_or_else(|| panic!("checked branch loses a live local"));
                if matches!(ty, SemanticType::Tensor(_)) {
                    let then_origin = self.region_descriptor_origin(
                        then_region,
                        then_value,
                        &mut BTreeSet::new(),
                    );
                    let else_origin = self.region_descriptor_origin(
                        else_region,
                        else_value,
                        &mut BTreeSet::new(),
                    );
                    if then_origin.is_some()
                        && then_origin
                            == self.region_descriptor_origin(
                                then_region,
                                *then_parameter,
                                &mut BTreeSet::new(),
                            )
                        && else_origin.is_some()
                        && else_origin
                            == self.region_descriptor_origin(
                                else_region,
                                *else_parameter,
                                &mut BTreeSet::new(),
                            )
                    {
                        // Both arms mutate the captured storage in place. The
                        // child semantic events carry the mutation; no tensor
                        // SSA value must cross the control-flow join.
                        continue;
                    }
                }
                changed.push((*ordinal, then_value, else_value));
            }
        }
        let mut then_results = Vec::new();
        let mut else_results = Vec::new();
        let mut output_types = Vec::new();
        for (_, then_value, else_value) in &changed {
            let ty = self.values[then_value.index()].ty.clone();
            self.flatten_call_argument(
                then_region,
                *then_value,
                &ty,
                &mut then_results,
                condition.span,
            );
            self.flatten_call_argument(
                else_region,
                *else_value,
                &ty,
                &mut else_results,
                condition.span,
            );
            Self::flatten_call_type(&ty, &mut output_types);
        }
        self.region_mut(then_region).results = then_results;
        self.region_mut(else_region).results = else_results;
        self.locals = before;
        let mut inputs = vec![condition_value];
        inputs.extend(captured_values);
        let outputs = self.emit(
            parent_region,
            NodeKind::If {
                then: then_region,
                otherwise: else_region,
            },
            inputs,
            output_types,
            condition.span,
        );
        let mut outputs = outputs.into_iter();
        for (ordinal, then_value, _) in changed {
            let ty = self.values[then_value.index()].ty.clone();
            let value =
                self.reconstruct_call_value(parent_region, &ty, &mut outputs, condition.span);
            self.locals[ordinal] = Some(value);
        }
        for (local, symbol) in join_symbols {
            let value = self.local(*local);
            let actual = self.runtime_int(value);
            self.symbols.insert(*symbol, actual);
        }
        assert!(
            outputs.next().is_none(),
            "branch join left an unbound result leaf"
        );
        Ok(())
    }

    fn lower_loop(
        &mut self,
        parent_region: RegionId,
        kind: ir::LoopKind,
        binder: ir::LocalId,
        start: &ir::Expr,
        end: &ir::Expr,
        body: &ir::Block,
        value_symbols: &[(ir::LocalId, SymbolId, SymbolId)],
        initialization: &crate::initialization::LoopInitialization,
    ) -> Result<(), SourceDiagnostic> {
        let start_value = self.lower_expr(parent_region, start)?;
        let end_value = self.lower_expr(parent_region, end)?;
        // Scheduling consumes the values that source execution actually
        // produced. Checked symbolic formulas establish bounds, but a typed
        // word expression may have wrapped before reaching this loop.
        let start_expression = self.runtime_int(start_value);
        let end_expression = self.runtime_int(end_value);
        let start_value = self.loop_index(parent_region, start_value, start_expression, start.span);
        let end_value = self.loop_index(parent_region, end_value, end_expression, end.span);
        let before = self.locals.clone();
        let parent_symbols = self.symbols.clone();
        let body_region = self.reserve_region(RegionKind::Root);
        let predecessor = self
            .last_event
            .get(&parent_region)
            .cloned()
            .unwrap_or_default();
        self.last_event.insert(body_region, predecessor);
        let binder_source = self.definition.body.locals[binder.index()].clone();
        let binder_ty = self.semantic_type(&binder_source.ty);
        let binder_value = self.region_parameter(body_region, binder_ty, binder_source.span);
        let (expression_binder, binder_symbol, binder_int) = self.builder.arena.loop_binder();
        if let Some(source_symbol) = self.definition.body.locals[binder.index()].symbol {
            self.symbols.insert(source_symbol, binder_int);
        }
        let mut child = before.clone();
        child[binder.index()] = Some(binder_value);
        let mut captures = Vec::new();
        for (ordinal, value) in before.iter().copied().enumerate() {
            if ordinal == binder.index() {
                continue;
            }
            if let Some(value) = value {
                let ty = self.values[value.index()].ty.clone();
                let span = self.definition.body.locals[ordinal].span;
                let mut parent_leaves = Vec::new();
                self.flatten_call_argument(parent_region, value, &ty, &mut parent_leaves, span);
                let mut parameter_leaves = Vec::new();
                let parameter =
                    self.region_parameter_tree(body_region, &ty, span, &mut parameter_leaves);
                assert_eq!(parent_leaves.len(), parameter_leaves.len());
                child[ordinal] = Some(parameter);
                captures.push((
                    ordinal,
                    value,
                    ty,
                    parent_leaves,
                    parameter,
                    parameter_leaves,
                ));
            }
        }
        for (local, header, _) in value_symbols {
            let parameter = child[local.index()]
                .expect("checked loop capture has no body parameter");
            let actual = self.runtime_int(parameter);
            self.symbols.insert(*header, actual);
        }
        fn leaf_paths(ty: &SemanticType, path: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
            match ty {
                SemanticType::Tuple(parts) => {
                    for (index, part) in parts.iter().enumerate() {
                        path.push(index);
                        leaf_paths(part, path, result);
                        path.pop();
                    }
                }
                SemanticType::Void => {}
                _ => result.push(path.clone()),
            }
        }
        let path_offset = self.definition.params.len();
        let mut transfer_leaves = vec![(path_offset + binder.index(), Vec::new(), 0)];
        let mut captured_values = captures
            .iter()
            .flat_map(|(_, _, _, values, _, _)| values.iter().copied())
            .collect::<Vec<_>>();
        let mut ordinal = 1;
        for (local, _, ty, _, _, parameters) in &captures {
            let mut paths = Vec::new();
            leaf_paths(ty, &mut Vec::new(), &mut paths);
            assert_eq!(paths.len(), parameters.len());
            for path in paths {
                transfer_leaves.push((path_offset + *local, path, ordinal));
                ordinal += 1;
            }
        }
        // A source formal may have been reassigned before this loop. Its
        // original value remains an actual capture when the retained transfer
        // refers to it; never substitute the current local for that value.
        let original_parameters = self.parameters.clone();
        let parent_formals = self.formals.clone();
        let mut child_formals = Vec::with_capacity(parent_formals.len());
        for (parameter, value) in original_parameters.into_iter().zip(&parent_formals) {
            let ordinal =
                if let Some(index) = captured_values.iter().position(|capture| capture == value) {
                    index + 1
                } else {
                    let ty = self.values[value.index()].ty.clone();
                    let span = self.values[value.index()].span;
                    self.region_parameter(body_region, ty, span);
                    captured_values.push(*value);
                    captured_values.len()
                };
            child_formals.push(self.region_mut(body_region).parameters[ordinal]);
            if let FunctionParameterOrigin::Source { ordinal: source, path } = parameter.origin {
                transfer_leaves.push((
                    source as usize,
                    path.iter().map(|p| *p as usize).collect(),
                    ordinal,
                ));
            }
        }
        let symbols = &self.symbols;
        let transferred_initialization = initialization.remap(
            &self.definition.arena,
            &mut self.builder.arena,
            &transfer_leaves,
            &mut |symbol, _| {
                AnyExpr::Int(
                    *symbols
                        .get(&symbol)
                        .expect("loop transfer symbol belongs to its checked scope"),
                )
            },
        );
        self.region_mut(body_region).initialization = Some(transferred_initialization);
        self.region_mut(body_region).kind = RegionKind::LoopBody {
            binder: BinderId::new(body_region, 0),
            expression_binder,
            binder_symbol,
            binder_value,
        };
        self.formals = child_formals;
        self.locals = child;
        let parallel = matches!(kind, ir::LoopKind::Independent);
        if parallel {
            self.parallel_depth = self
                .parallel_depth
                .checked_add(1)
                .unwrap_or_else(|| panic!("parallel loop nesting exceeds u32::MAX"));
            let participant = BinderId::new(body_region, 0);
            self.parallel_participants.push(participant);
            assert!(self
                .participant_bindings
                .insert(binder, participant)
                .is_none());
        }
        let returned = self.lower_block(body_region, body);
        if parallel {
            let popped = self
                .parallel_participants
                .pop()
                .unwrap_or_else(|| panic!("parallel participant stack is unbalanced"));
            assert_eq!(popped, BinderId::new(body_region, 0));
            assert_eq!(
                self.participant_bindings.remove(&binder),
                Some(BinderId::new(body_region, 0))
            );
            self.parallel_depth -= 1;
        }
        let returned = returned?;
        self.symbols = parent_symbols;
        assert!(
            returned.is_empty(),
            "checked loop body contains an early return"
        );
        let after = self.locals.clone();
        self.formals = parent_formals;
        let mut carries = Vec::new();
        let mut output_types = Vec::new();
        let mut yielded = Vec::new();
        let all_capture_values = captured_values;
        let mut final_leaf_sources = Vec::new();
        self.locals = before;
        for (ordinal, _, ty, initial_leaves, _parameter, parameter_leaves) in &captures {
            let next = after[*ordinal].unwrap_or_else(|| panic!("checked loop loses a live local"));
            let mut next_leaves = Vec::new();
            self.flatten_call_argument(
                body_region,
                next,
                ty,
                &mut next_leaves,
                self.definition.body.locals[*ordinal].span,
            );
            assert_eq!(next_leaves.len(), parameter_leaves.len());
            let mut source_indices = Vec::with_capacity(next_leaves.len());
            for ((initial, parameter), next) in
                initial_leaves.iter().zip(parameter_leaves).zip(next_leaves)
            {
                if next == *parameter {
                    source_indices.push(None);
                    continue;
                }
                // Element/store updates change the contents of a place, not
                // the tensor value identifying that place. This holds for both
                // ordered and parallel loops; only actual value rebinding is
                // represented by a loop carry. Leaf write events retain the
                // mutation ordering and authority.
                let next_storage =
                    self.region_descriptor_origin(body_region, next, &mut BTreeSet::new());
                let parameter_storage =
                    self.region_descriptor_origin(body_region, *parameter, &mut BTreeSet::new());
                let aliases_parameter_storage =
                    matches!(self.values[next.index()].ty, SemanticType::Tensor(_))
                        && next_storage.is_some()
                        && next_storage == parameter_storage;
                if aliases_parameter_storage {
                    source_indices.push(None);
                    continue;
                }
                if parallel {
                    return Err(SourceDiagnostic::new(
                        &self.module.sources.files()[self.definition.file],
                        self.definition.body.locals[*ordinal].span,
                        crate::checked::DiagnosticRule::Type,
                        "a `parallel for` body cannot carry reassigned state; use an explicit reduction or write through a disjoint tensor view".to_owned(),
                    ));
                }
                let result = SemanticValueId::new(
                    self.id,
                    u32::try_from(self.values.len() + output_types.len())
                        .expect("function has more than u32::MAX values"),
                );
                carries.push(Carry {
                    initial: *initial,
                    parameter: *parameter,
                    yielded: next,
                    result,
                });
                yielded.push(next);
                output_types.push(self.values[next.index()].ty.clone());
                source_indices.push(Some(output_types.len() - 1));
            }
            final_leaf_sources.push((*ordinal, ty.clone(), initial_leaves.clone(), source_indices));
        }
        self.region_mut(body_region).results = yielded;
        let mut inputs = vec![start_value, end_value];
        inputs.extend(all_capture_values.iter().copied());
        let semantic_kind = match kind {
            ir::LoopKind::Ordered => LoopKind::Ordered,
            ir::LoopKind::Independent => LoopKind::Parallel,
        };
        let outputs = self.emit(
            parent_region,
            NodeKind::Loop {
                kind: semantic_kind,
                body: body_region,
                carries: carries.clone(),
            },
            inputs,
            output_types,
            start.span,
        );
        for (carry, output) in carries.iter().zip(&outputs) {
            assert_eq!(
                carry.result, *output,
                "loop carry result allocation diverged from node outputs"
            );
        }
        for (ordinal, ty, initial, sources) in final_leaf_sources {
            let leaves = sources
                .into_iter()
                .enumerate()
                .map(|(leaf, source)| source.map_or(initial[leaf], |output| outputs[output]))
                .collect::<Vec<_>>();
            let mut leaves = leaves.into_iter();
            let value = self.reconstruct_call_value(parent_region, &ty, &mut leaves, start.span);
            assert!(
                leaves.next().is_none(),
                "loop join left an unbound result leaf"
            );
            self.locals[ordinal] = Some(value);
        }
        for (local, _, exit) in value_symbols {
            let value = self.local(*local);
            let actual = self.runtime_int(value);
            self.symbols.insert(*exit, actual);
        }
        Ok(())
    }

    /// Crosses the checked scalar/index boundary for loop scheduling.
    ///
    /// A semantic loop is scheduled over the non-negative index domain. The
    /// source endpoint may be a mathematical quantity or a typed word; its
    /// actual semantic SSA value is the only valid input to this conversion.
    fn loop_index(
        &mut self,
        region: RegionId,
        scalar: SemanticValueId,
        expression: IntExpr,
        span: crate::span::Span,
    ) -> SemanticValueId {
        let value = self.builder.arena.nat_from_int(expression);
        let one = self.builder.arena.nat(1);
        let bound = self.builder.arena.nat_add(value, one);
        self.emit(
            region,
            NodeKind::Primitive(PrimitiveId::Symbolic(expression)),
            vec![scalar],
            vec![SemanticType::Index { bound }],
            span,
        )[0]
    }
}
