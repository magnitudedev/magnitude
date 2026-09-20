use seismic_lang::{
    family::{self, Workload},
    logical::{
        self, FragmentId, LogicalBlock, LogicalExpr, LogicalExprKind, LogicalOperationKind,
        StorageOrigin, Type, ValueKind,
    },
    program::{compile, SourceFile},
};

fn specialize_entry(source: &str, entry: &str, shapes: &[(&str, i64)]) -> logical::LogicalProgram {
    let program = compile(&[SourceFile {
        path: "planning-boundary.seismic".into(),
        text: source.into(),
    }])
    .unwrap_or_else(|diagnostics| {
        panic!(
            "{}",
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.render())
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    let workload = Workload {
        shapes: shapes
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect(),
        ..Workload::default()
    };
    let supports = |_: &seismic_lang::sir::IntrinsicUse| Ok(());
    let environment = family::TargetEnvironment {
        target: "cpu",
        capability_fingerprint: "planning-boundary-tests-v1",
        supports_intrinsic: &supports,
    };
    logical::specialize_entry_contract(&program, entry, &environment, &workload)
        .expect("logical entry specialization")
}

fn logical_exprs<'a>(block: &'a LogicalBlock, out: &mut Vec<&'a LogicalExpr>) {
    fn expression<'a>(value: &'a LogicalExpr, out: &mut Vec<&'a LogicalExpr>) {
        out.push(value);
        match &value.kind {
            LogicalExprKind::Tuple(values)
            | LogicalExprKind::Math { args: values, .. }
            | LogicalExprKind::Intrinsic { args: values, .. } => {
                for value in values {
                    expression(value, out);
                }
            }
            LogicalExprKind::Range(a, b) | LogicalExprKind::Binary { lhs: a, rhs: b, .. } => {
                expression(a, out);
                expression(b, out);
            }
            LogicalExprKind::Field(value, _)
            | LogicalExprKind::View { base: value, .. }
            | LogicalExprKind::Snapshot { source: value, .. }
            | LogicalExprKind::Decode { source: value, .. }
            | LogicalExprKind::Materialize { value, .. }
            | LogicalExprKind::Cast { expr: value, .. }
            | LogicalExprKind::Unary { expr: value, .. }
            | LogicalExprKind::Reduce { value, .. }
            | LogicalExprKind::Extent { base: value, .. }
            | LogicalExprKind::Accessor { base: value, .. }
            | LogicalExprKind::Geometry { base: value, .. } => expression(value, out),
            LogicalExprKind::Filled { like, .. } => expression(like, out),
            LogicalExprKind::Index { base, .. } => expression(base, out),
            LogicalExprKind::Select { cond, then, els } => {
                expression(cond, out);
                expression(then, out);
                expression(els, out);
            }
            LogicalExprKind::Call { args, .. } => {
                for value in args {
                    expression(value, out);
                }
            }
            LogicalExprKind::Atomic { place, value, .. } => {
                expression(place, out);
                expression(value, out);
            }
            LogicalExprKind::Region(_)
            | LogicalExprKind::Int(_)
            | LogicalExprKind::Float(_)
            | LogicalExprKind::Bool(_)
            | LogicalExprKind::Value(_)
            | LogicalExprKind::Shape(_)
            | LogicalExprKind::Construct { .. }
            | LogicalExprKind::Coordinate(_) => {}
        }
    }
    for operation in block {
        match &operation.kind {
            LogicalOperationKind::Bind { value, .. } | LogicalOperationKind::Expr(value) => {
                expression(value, out)
            }
            LogicalOperationKind::Assign { target, value, .. }
            | LogicalOperationKind::Publish {
                value,
                destination: target,
            } => {
                expression(target, out);
                expression(value, out);
            }
            LogicalOperationKind::For {
                lo,
                hi,
                source,
                body,
                ..
            } => {
                expression(lo, out);
                expression(hi, out);
                if let Some(source) = source {
                    expression(source, out);
                }
                logical_exprs(body, out);
            }
            LogicalOperationKind::Coordinates { value, body, .. } => {
                expression(value, out);
                logical_exprs(body, out);
            }
            LogicalOperationKind::Members { body, .. } => logical_exprs(body, out),
            LogicalOperationKind::If {
                condition,
                then,
                els,
            } => {
                expression(condition, out);
                logical_exprs(then, out);
                logical_exprs(els, out);
            }
            LogicalOperationKind::Yield(values) => {
                for value in values {
                    expression(value, out);
                }
            }
            LogicalOperationKind::Return(writes) => {
                for write in writes {
                    expression(&write.value, out);
                }
            }
            LogicalOperationKind::Stages(stages) => {
                for stage in stages {
                    logical_exprs(&stage.body, out);
                }
            }
            LogicalOperationKind::Region(_) => {}
        }
    }
}

#[test]
fn logical_results_exist_before_any_witness_or_physical_plan() {
    let logical = specialize_entry(
        "fn nested[N](a: tensor[N] f32) -> (tensor[N] f32, (tensor[N] f32, tensor[N] f32)):\n    return a, (a, a)\n",
        "nested",
        &[("N", 4)],
    );

    assert_eq!(
        logical
            .result_slots
            .iter()
            .map(|slot| slot.path.clone())
            .collect::<Vec<_>>(),
        [vec![0], vec![1, 0], vec![1, 1]]
    );
    assert!(logical.result_slots.iter().all(|slot| {
        matches!(
            &logical.storage[slot.storage.0 as usize].origin,
            StorageOrigin::Result { path } if path == &slot.path
        )
    }));
    assert_eq!(logical.choices.len(), 1);
    assert_eq!(logical.choices[0].alternatives.len(), 1);
    logical.verify().expect("specialized logical contract");
}

#[test]
fn scalar_results_are_values_without_fabricated_result_storage() {
    let logical = specialize_entry(
        "fn scalar[N](t: tensor[N] f32) -> f32:\n    return reduce(f32(t), 0, sum)\n",
        "scalar",
        &[("N", 4)],
    );

    assert!(logical.result_slots.is_empty());
    assert_eq!(
        logical.choices[0].interface.results,
        [Type::Scalar(seismic_lang::types::DType::F32)]
    );
}

#[test]
fn range_parameters_are_explicit_logical_values() {
    let logical = specialize_entry(
        "fn bounded[N](selected: range[N]):\n    return\n",
        "bounded",
        &[("N", 4)],
    );

    assert!(logical
        .values
        .iter()
        .any(|value| matches!(value.kind, ValueKind::RangeParameter { ordinal: 0, .. })));
}

#[test]
fn repeated_callee_loop_extents_keep_occurrence_specific_capacities() {
    let logical = specialize_entry(
        "fn rows[M, N](x: &tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut out = tensor[M, N] f32\n    parallel for row in 0..M:\n        out[row:row + 1] = x[row:row + 1]\n    return out\n\nfn paired[A, B, N](x: &tensor[A, N] f32, y: &tensor[B, N] f32) -> (tensor[A, N] f32, tensor[B, N] f32):\n    return rows(x), rows(y)\n",
        "paired",
        &[("A", 4), ("B", 7), ("N", 3)],
    );

    let row_bounds = logical
        .extent_bounds
        .iter()
        .filter(|(symbol, _)| symbol.ends_with(".var2"))
        .map(|(symbol, bound)| (symbol.clone(), bound.as_constant()))
        .collect::<Vec<_>>();
    assert_eq!(row_bounds.len(), 2);
    assert!(row_bounds
        .iter()
        .all(|(symbol, _)| symbol.starts_with("@runtime.choice")));
    assert_eq!(
        row_bounds
            .iter()
            .map(|(_, bound)| *bound)
            .collect::<std::collections::BTreeSet<_>>(),
        [Some(3), Some(6)].into_iter().collect()
    );
    for (symbol, bound) in row_bounds {
        assert_eq!(
            logical.extent_capacity(
                &seismic_lang::sym::Sym::param(&symbol).add(&seismic_lang::sym::Sym::constant(1))
            ),
            bound.map(|bound| bound + 1)
        );
    }
}

#[test]
fn every_applicable_body_is_a_branch_local_fragment_before_selection() {
    let logical = specialize_entry(
        "fn copy[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nlower copy[N](x: &tensor[N] f32) -> tensor[N] f32 for cpu:\n    return to_owned(x)\n",
        "copy",
        &[("N", 8)],
    );

    assert_eq!(logical.choices.len(), 1);
    assert_eq!(logical.choices[0].alternatives.len(), 2);
    assert_eq!(logical.fragments.len(), 2);
    assert_ne!(
        logical.choices[0].alternatives[0].fragment,
        logical.choices[0].alternatives[1].fragment
    );
    assert!(logical
        .fragments
        .iter()
        .all(|fragment| { fragment.choice == logical.entry_choice && !fragment.body.is_empty() }));
    logical
        .verify()
        .expect("complete choice-bearing logical program");
}

#[test]
fn nested_calls_name_nested_choices_without_embedding_a_witness() {
    let logical = specialize_entry(
        "fn helper[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nlower helper[N](x: &tensor[N] f32) -> tensor[N] f32 for cpu:\n    return to_owned(x)\n\nfn caller[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return helper(x)\n",
        "caller",
        &[("N", 4)],
    );

    assert_eq!(logical.choices.len(), 2);
    assert_eq!(logical.choices[1].alternatives.len(), 2);
    let root = &logical.fragments[logical.choices[0].alternatives[0].fragment.0 as usize];
    let mut expressions = Vec::new();
    logical_exprs(&root.body, &mut expressions);
    let call = expressions
        .iter()
        .find_map(|expression| match &expression.kind {
            LogicalExprKind::Call {
                choice, results, ..
            } if *choice == logical::ChoiceId(1) => Some(results),
            _ => None,
        })
        .expect("nested call expression");
    assert_eq!(call.len(), 1);
    assert_eq!(call[0].path, Vec::<u32>::new());
    assert!(matches!(
        root.storage[call[0].storage.0 as usize].origin,
        logical::LocalStorageOrigin::CallResult {
            choice: logical::ChoiceId(1),
            ref path,
        } if path.is_empty()
    ));
    logical.verify().expect("nested choice graph");
}

#[test]
fn logical_loops_preserve_authored_ordering_not_physical_mapping() {
    let logical = specialize_entry(
        "fn loops[N](x: &tensor[N] f32, output: tensor[N] f32) -> tensor[N] f32:\n    let mut result = output\n    parallel for i in 0..N:\n        result[i] = x[i]\n    for i in 0..N:\n        result[i] = result[i] + 1.0\n    return result\n",
        "loops",
        &[("N", 4)],
    );
    let body = &logical.fragments[0].body;
    let kinds = body
        .iter()
        .filter_map(|operation| match operation.kind {
            LogicalOperationKind::For { independent, .. } => Some(independent),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(kinds, [true, false]);
    logical.verify().expect("logical loop semantics");
}

#[test]
fn owned_returns_complete_destination_ports_by_transfer() {
    let logical = specialize_entry(
        "fn pair[N](x: &tensor[N] f32) -> (tensor[N] f32, f32):\n    return to_owned(x), reduce(f32(x), 0, sum)\n",
        "pair",
        &[("N", 4)],
    );
    let returns = logical.fragments[0]
        .body
        .iter()
        .find_map(|operation| match &operation.kind {
            LogicalOperationKind::Return(writes) => Some(writes),
            _ => None,
        })
        .expect("logical return");
    assert_eq!(returns.len(), 2);
    assert!(returns[0].transfer);
    assert!(!returns[1].transfer);
    assert_eq!(returns[0].path, [0]);
    assert_eq!(returns[1].path, [1]);
    logical.verify().expect("destination-passing return");
}

#[test]
fn verifier_rejects_cross_branch_fragment_identity() {
    let mut logical = specialize_entry(
        "fn copy[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nlower copy[N](x: &tensor[N] f32) -> tensor[N] f32 for cpu:\n    return to_owned(x)\n",
        "copy",
        &[("N", 8)],
    );
    logical.choices[0].alternatives[1].fragment = FragmentId(0);
    assert!(logical
        .verify()
        .unwrap_err()
        .contains("shared by alternatives"));
}
