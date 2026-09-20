//! Targeted tests: portable legalization matrix coverage over
//! the registry, obligation discharge proof-vs-check classification,
//! numerical composition and strict-policy retention, reduction strategy
//! admission, and formation over constructed logical programs.

use super::*;
use seismic_lang::{
    intrinsics::{primitives, PrimitiveId, ReduceOp},
    logical::{
        construct, ChoiceId, EffectiveTargetIdentity, GraphValueId, LogicalNodeKind, ReductionNode,
        ReductionOrder, RuntimeExtent, SafetyObligation,
    },
    precision::{Limit, NumericalAssessment, PrecisionPolicy, Tolerance},
    program::{compile, SourceFile},
    sir::IntrinsicUse,
    span::Span,
    types::{DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType, ValueType},
};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check(sources: &[(&str, &str)]) -> Result<seismic_lang::sir::Program, String> {
    let files: Vec<SourceFile> = sources
        .iter()
        .map(|(path, text)| SourceFile {
            path: path.to_string(),
            text: text.to_string(),
        })
        .collect();
    compile(&files).map_err(|d| d.iter().map(|d| d.render()).collect::<Vec<_>>().join("\n"))
}

fn supports_all(_: &IntrinsicUse) -> Result<(), String> {
    Ok(())
}

fn target(backend: &str) -> EffectiveTargetIdentity {
    EffectiveTargetIdentity {
        backend: backend.to_string(),
        capability_fingerprint: "test-fingerprint".to_string(),
    }
}

fn shapes(entries: &[(&str, i64)]) -> BTreeMap<String, i64> {
    entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn entry_graph(
    sources: &[(&str, &str)],
    entry: &str,
    backend: &str,
    shape_values: &[(&str, i64)],
) -> seismic_lang::logical::LogicalProgram {
    let program = check(sources).expect("the sources check");
    let logical = construct(
        &program,
        entry,
        &target(backend),
        &supports_all,
        shapes(shape_values),
        BTreeMap::new(),
    )
    .expect("construction succeeds");
    logical.verify().expect("the built program verifies");
    logical
}

fn dense(elem: DType, axes: Vec<ExtentExpr>) -> ValueType {
    ValueType::Tensor(TensorType::new(axes, Elem::Dtype(elem)))
}

fn packed() -> ValueType {
    ValueType::Tensor(TensorType::new(
        vec![ExtentExpr::Static(8)],
        Elem::Repr("q4g64".into()),
    ))
}

fn scalar(dtype: DType) -> ValueType {
    ValueType::Scalar(dtype)
}

/// Operand types sufficient to classify each registry primitive family.
fn sample_operands(id: &PrimitiveId) -> Vec<ValueType> {
    use PrimitiveId::*;
    let f32_tensor = || dense(DType::F32, vec![ExtentExpr::Static(4)]);
    match id {
        TuplePack => vec![scalar(DType::I32), scalar(DType::F32)],
        TupleGet(_) => vec![ValueType::Tuple(
            NonEmpty::new(vec![scalar(DType::I32), scalar(DType::F32)]).unwrap(),
        )],
        RangeMake => vec![scalar(DType::I32), scalar(DType::I32)],
        RangeStart | RangeEnd => vec![ValueType::Range {
            bound: ExtentExpr::Static(8),
        }],
        Unary(_) | Math(_) | Cast(_) => vec![scalar(DType::F32)],
        Binary(_) => vec![scalar(DType::F32), scalar(DType::F32)],
        Select => vec![scalar(DType::Bool), scalar(DType::F32), scalar(DType::F32)],
        TensorAlloc { .. } | Fill { .. } => vec![f32_tensor()],
        Materialize
        | Clone
        | Load
        | Transpose
        | Reshape
        | SliceView { .. }
        | Extent { .. }
        | ValidExtent { .. } => vec![f32_tensor()],
        ElementRead { .. } => vec![f32_tensor()],
        ElementWrite { .. } => vec![f32_tensor()],
        CopyInto => vec![f32_tensor(), f32_tensor()],
        Decode | PackedRead(_) => vec![packed()],
        Atomic { .. } => vec![f32_tensor(), scalar(DType::F32)],
        Reduce { .. } => vec![dense(DType::F32, vec![ExtentExpr::Static(4)])],
    }
}

// ---------------------------------------------------------------------------
// 16.1 matrix coverage over the registry
// ---------------------------------------------------------------------------

#[test]
fn matrix_covers_every_registry_primitive() {
    let signatures = primitives();
    assert!(
        signatures.len() > 60,
        "the registry enumerates its families"
    );
    for signature in &signatures {
        // Reduction rows are ReductionNodes: they are consumed by reduction
        // strategies and correctly refuse scalar legalization (asserted in
        // `reductions_never_legalize_as_scalar_primitives`).
        if matches!(signature.id, PrimitiveId::Reduce { .. }) {
            continue;
        }
        let operands = sample_operands(&signature.id);
        match universal_form(
            &seismic_lang::logical::PrimitiveOp::Primitive(signature.id.clone()),
            &operands,
        ) {
            // The universal column is never optional-Inapplicable for a
            // primitive: every family legalizes.
            Ok(UniversalLegalization::Form(form)) => {
                assert_eq!(universal_numerical(&form), NumericalTransfer::Exact);
            }
            other => panic!(
                "primitive `{}` failed universal legalization: {other:?}",
                signature.id.name()
            ),
        }
    }
}

#[test]
fn reductions_never_legalize_as_scalar_primitives() {
    let error = universal_form(
        &seismic_lang::logical::PrimitiveOp::Primitive(PrimitiveId::Reduce {
            op: ReduceOp::Sum,
            axis: 0,
            unordered: false,
        }),
        &sample_operands(&PrimitiveId::Reduce {
            op: ReduceOp::Sum,
            axis: 0,
            unordered: false,
        }),
    )
    .expect_err("a reduction is a ReductionNode, never a scalar primitive");
    assert!(error.0.contains("reduction"));
}

#[test]
fn capabilities_have_no_portable_opcode() {
    let intrinsic = seismic_lang::intrinsics::IntrinsicId {
        capability: seismic_lang::intrinsics::CapabilityId::new("metal", "subgroup"),
        name: "simd_sum".into(),
    };
    match universal_form(
        &seismic_lang::logical::PrimitiveOp::Capability(intrinsic.clone()),
        &[],
    ) {
        Ok(UniversalLegalization::RequiresCapability { intrinsic: found }) => {
            assert_eq!(found, intrinsic);
        }
        other => panic!("expected the capability route, got {other:?}"),
    }
}

#[test]
fn arithmetic_classifies_by_dtype_and_operator() {
    let f = |l: ValueType, r: ValueType, id: PrimitiveId| {
        universal_form(&seismic_lang::logical::PrimitiveOp::Primitive(id), &[l, r])
            .expect("binary arithmetic legalizes")
    };
    use seismic_lang::syntax::ast::BinaryOp::*;
    let int = || scalar(DType::I32);
    let flt = || scalar(DType::F32);
    // Float arithmetic at the registry dtype and rounding point.
    for op in [Add, Sub, Mul, Div, Rem] {
        assert_eq!(
            f(flt(), flt(), PrimitiveId::Binary(op)),
            UniversalLegalization::Form(UniversalForm::FloatArithmetic { dtype: DType::F32 })
        );
    }
    // Integer arithmetic as 32-bit wrapping operations (planned checks ride
    // on the node's obligations, not the form).
    for op in [Add, Sub, Mul, Div, Rem, Shl, Shr, BitAnd, BitOr, BitXor] {
        assert_eq!(
            f(int(), int(), PrimitiveId::Binary(op)),
            UniversalLegalization::Form(UniversalForm::IntegerArithmetic)
        );
    }
    // Comparisons and logic are typed SSA.
    for op in [Eq, Ne, Lt, Le, Gt, Ge, And, Or] {
        assert_eq!(
            f(int(), int(), PrimitiveId::Binary(op)),
            UniversalLegalization::Form(UniversalForm::TypedSsa)
        );
    }
    // Unary classification.
    let unary = |op, operand| {
        universal_form(
            &seismic_lang::logical::PrimitiveOp::Primitive(PrimitiveId::Unary(op)),
            &[operand],
        )
        .expect("unary legalizes")
    };
    assert_eq!(
        unary(seismic_lang::syntax::ast::UnaryOp::Neg, scalar(DType::F16)),
        UniversalLegalization::Form(UniversalForm::FloatArithmetic { dtype: DType::F16 })
    );
    assert_eq!(
        unary(seismic_lang::syntax::ast::UnaryOp::Neg, scalar(DType::I32)),
        UniversalLegalization::Form(UniversalForm::IntegerArithmetic)
    );
    assert_eq!(
        unary(seismic_lang::syntax::ast::UnaryOp::Not, scalar(DType::Bool)),
        UniversalLegalization::Form(UniversalForm::TypedSsa)
    );
    assert_eq!(
        unary(
            seismic_lang::syntax::ast::UnaryOp::BitNot,
            scalar(DType::I32)
        ),
        UniversalLegalization::Form(UniversalForm::IntegerArithmetic)
    );
}

#[test]
fn transcendental_reference_is_the_one_versioned_algorithm() {
    assert_eq!(SEISMIC_MATH.identity, "seismic_math");
    assert_eq!(SEISMIC_MATH_VERSION, 1);
    registry_math_is_versioned().expect("registry math is versioned seismic_math");
    for op in seismic_lang::intrinsics::math_ops() {
        match universal_form(
            &seismic_lang::logical::PrimitiveOp::Primitive(PrimitiveId::Math(op)),
            &[scalar(DType::F32)],
        ) {
            Ok(UniversalLegalization::Form(UniversalForm::SoftwareMath { reference, .. })) => {
                assert_eq!(reference, SEISMIC_MATH);
            }
            other => panic!(
                "math `{}` did not legalize as software math: {other:?}",
                op.name()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Obligation discharge classification
// ---------------------------------------------------------------------------

fn index_value(id: u32, bound: ExtentExpr) -> (GraphValueId, ValueType) {
    (GraphValueId(id), ValueType::Index { bound })
}

#[test]
fn obligations_classify_into_proofs_and_checks() {
    let mut facts = GraphFacts::default();
    let (refined, refined_ty) = index_value(0, ExtentExpr::Static(4));
    facts.types.insert(refined, refined_ty);
    let (wide, wide_ty) = index_value(1, ExtentExpr::Static(16));
    facts.types.insert(wide, wide_ty);
    facts
        .types
        .insert(GraphValueId(2), ValueType::Scalar(DType::I32));
    facts.constants.insert(GraphValueId(2), 3);
    facts
        .types
        .insert(GraphValueId(3), ValueType::Scalar(DType::I32));
    facts.constants.insert(GraphValueId(3), -1);
    // A runtime dividend with a -1 divisor still needs the overflow check.
    facts
        .types
        .insert(GraphValueId(4), ValueType::Scalar(DType::I32));
    let span = Span::default();

    // An index-refined value inside a wider extent is statically proved.
    assert_eq!(
        discharge(
            &SafetyObligation::IndexInBounds {
                index: refined,
                extent: ExtentExpr::Static(8)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::IndexRefinement {
            bound: ExtentExpr::Static(4),
            extent: ExtentExpr::Static(8),
        })
    );
    // A refinement wider than the extent proves nothing: runtime check.
    match discharge(
        &SafetyObligation::IndexInBounds {
            index: refined,
            extent: ExtentExpr::Static(2),
        },
        &facts,
        span,
    ) {
        ObligationDischarge::RuntimeChecked(check) => {
            assert!(matches!(
                check.predicate,
                CheckPredicate::IndexInBounds { .. }
            ));
            assert_eq!(check.inactive, InactiveBehavior::SkipOperation);
            assert_eq!(check.status.kind, SafetyKind::IndexOutOfBounds);
        }
        other => panic!("expected a runtime check, got {other:?}"),
    }
    // A constant index inside a static extent is proved; outside it, the
    // alternative is statically impossible.
    assert_eq!(
        discharge(
            &SafetyObligation::IndexInBounds {
                index: GraphValueId(2),
                extent: ExtentExpr::Static(8)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::ConstantValue { value: 3 })
    );
    assert!(matches!(
        discharge(
            &SafetyObligation::IndexInBounds {
                index: GraphValueId(2),
                extent: ExtentExpr::Static(2)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyImpossible(_)
    ));
    // Division/shift constants.
    assert_eq!(
        discharge(
            &SafetyObligation::DivisorNonZero {
                value: GraphValueId(2)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::ConstantValue { value: 3 })
    );
    assert!(matches!(
        discharge(
            &SafetyObligation::ShiftInRange {
                value: GraphValueId(3)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyImpossible(_)
    ));
    assert_eq!(
        discharge(
            &SafetyObligation::ShiftInRange {
                value: GraphValueId(2)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::ConstantValue { value: 3 })
    );
    // A constant divisor of -1 with a runtime dividend needs the check; with
    // a constant dividend away from i32::MIN it is proved.
    assert!(matches!(
        discharge(
            &SafetyObligation::SignedDivisionNoOverflow {
                lhs: GraphValueId(4),
                rhs: GraphValueId(3)
            },
            &facts,
            span
        ),
        ObligationDischarge::RuntimeChecked(_)
    ));
    assert_eq!(
        discharge(
            &SafetyObligation::SignedDivisionNoOverflow {
                lhs: GraphValueId(2),
                rhs: GraphValueId(3)
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::ConstantValue { value: 3 })
    );
}

#[test]
fn shape_products_prove_statically_or_by_capacity() {
    let mut facts = GraphFacts::default();
    facts.runtime_extents.insert(
        RuntimeExtentId(0),
        RuntimeExtent {
            id: RuntimeExtentId(0),
            value: seismic_lang::logical::RuntimeScalarExpr::Const(4),
            capacity: 4,
            expected: None,
        },
    );
    let span = Span::default();
    // All-static factors: the checked product decides.
    assert_eq!(
        discharge(
            &SafetyObligation::ShapeProductFits {
                factors: vec![ExtentExpr::Static(3), ExtentExpr::Static(4)],
                bits: 32,
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::StaticProductFits {
            product: 12,
            bits: 32
        })
    );
    assert!(matches!(
        discharge(
            &SafetyObligation::ShapeProductFits {
                factors: vec![ExtentExpr::Static(1 << 31), ExtentExpr::Static(4)],
                bits: 32,
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyImpossible(_)
    ));
    // Runtime factors bounded by capacities that fit: proved.
    assert_eq!(
        discharge(
            &SafetyObligation::ShapeProductFits {
                factors: vec![
                    ExtentExpr::Runtime(RuntimeExtentId(0)),
                    ExtentExpr::Runtime(RuntimeExtentId(0)),
                ],
                bits: 32,
            },
            &facts,
            span
        ),
        ObligationDischarge::StaticallyProved(StaticProof::CapacityBounded {
            capacity_product: 16,
            bits: 32
        })
    );
}

// ---------------------------------------------------------------------------
// Numerical composition and policy
// ---------------------------------------------------------------------------

fn round(dtype: DType, count: CountExpr) -> NumericalTransfer {
    NumericalTransfer::Round { dtype, count }
}

#[test]
fn transfers_compose_analytically() {
    use NumericalTransfer::*;
    // Exact is the identity.
    assert_eq!(
        compose(&Exact, &round(DType::F32, CountExpr::Const(1))),
        round(DType::F32, CountExpr::Const(1))
    );
    // Round after round adds counts at the wider-error dtype.
    assert_eq!(
        compose(
            &round(DType::F32, CountExpr::Const(2)),
            &round(DType::BF16, CountExpr::Const(3))
        ),
        round(DType::BF16, CountExpr::Const(5))
    );
    // A bounded approximation absorbs a closed-form rounding count.
    let bound = seismic_lang::intrinsics::ErrorBound {
        relative: 1e-6,
        absolute: 0.0,
    };
    match compose(
        &Approximate {
            operation: PrimitiveId::Math(seismic_lang::intrinsics::MathOp::Exp),
            bound,
        },
        &round(DType::F32, CountExpr::Const(4)),
    ) {
        Approximate { bound, .. } => {
            assert!((bound.relative - (1e-6 + 4.0 * unit_roundoff(DType::F32))).abs() < 1e-18);
        }
        other => panic!("expected an absorbed bound, got {other:?}"),
    }
    // Reassociation subsumes countable rounding.
    let topology = ReductionTopology::SerialAxis {
        axis: 0,
        length: ExtentExpr::Static(8),
    };
    assert_eq!(
        compose(
            &Reassociate {
                op: ReduceOp::Sum,
                topology: topology.clone()
            },
            &round(DType::F32, CountExpr::Const(1))
        ),
        Reassociate {
            op: ReduceOp::Sum,
            topology
        }
    );
    // Unknown absorbs.
    assert!(matches!(
        compose(
            &Unknown {
                reason: "authored".into()
            },
            &round(DType::F32, CountExpr::Const(1))
        ),
        Unknown { .. }
    ));
    // Whole-candidate composition.
    assert_eq!(
        compose_all(&[
            Exact,
            round(DType::F32, CountExpr::Const(1)),
            round(DType::F32, CountExpr::Const(1))
        ]),
        round(DType::F32, CountExpr::Const(2))
    );
}

#[test]
fn strict_policies_retain_only_exact_transfers() {
    let resolve = |_: RuntimeExtentId| Some(64u64);
    // Exact transfer satisfies everything.
    for policy in [
        PrecisionPolicy::Exact,
        PrecisionPolicy::bounded(Tolerance::EXACT),
        PrecisionPolicy::Unconstrained,
    ] {
        assert_eq!(
            satisfies_policy(&NumericalTransfer::Exact, &policy, &resolve),
            PolicyDecision::Satisfied
        );
    }
    // Extra rounding is rejected by strict policies and admitted by bounded
    // policies when the analytical bound fits.
    assert!(matches!(
        satisfies_policy(
            &round(DType::F32, CountExpr::Const(1)),
            &PrecisionPolicy::Exact,
            &resolve
        ),
        PolicyDecision::Violated { .. }
    ));
    let bounded = PrecisionPolicy::bounded(Tolerance {
        absolute: Limit::ZERO,
        relative: Limit::new(1e-6).unwrap(),
        relative_floor: Limit::ZERO,
        ulps: None,
    });
    assert_eq!(
        satisfies_policy(&round(DType::F32, CountExpr::Const(1)), &bounded, &resolve),
        PolicyDecision::Satisfied
    );
    // A runtime-dependent count that would exceed the tolerance needs
    // evidence.
    match satisfies_policy(
        &round(DType::F16, CountExpr::Elements(ExtentExpr::Static(4096))),
        &bounded,
        &resolve,
    ) {
        PolicyDecision::RequiresEvidence { .. } => {}
        other => panic!("expected an evidence requirement, got {other:?}"),
    }
    // Reassociation and unknown transfers always need admission.
    let reassociate = NumericalTransfer::Reassociate {
        op: ReduceOp::Sum,
        topology: ReductionTopology::SerialAxis {
            axis: 0,
            length: ExtentExpr::Static(8),
        },
    };
    assert!(matches!(
        satisfies_policy(&reassociate, &PrecisionPolicy::Exact, &resolve),
        PolicyDecision::Violated { .. }
    ));
    assert!(matches!(
        satisfies_policy(&reassociate, &bounded, &resolve),
        PolicyDecision::RequiresEvidence { .. }
    ));
    assert!(matches!(
        satisfies_policy(
            &NumericalTransfer::Unknown {
                reason: "authored lowering".into()
            },
            &bounded,
            &resolve
        ),
        PolicyDecision::RequiresEvidence { .. }
    ));
    // Exploration admits everything.
    for transfer in [
        NumericalTransfer::Exact,
        round(DType::F32, CountExpr::Const(1)),
        reassociate,
        NumericalTransfer::Unknown { reason: "x".into() },
    ] {
        assert_eq!(
            satisfies_policy(&transfer, &PrecisionPolicy::Unconstrained, &resolve),
            PolicyDecision::Satisfied
        );
    }
}

#[test]
fn evidence_is_keyed_to_the_complete_identity() {
    let key = EvidenceKey::new(
        seismic_lang::logical::LogicalIdentity([0; 32]),
        WorkloadFingerprint([1; 32]),
        target("cpu"),
        ToolchainId("clang-19".into()),
        AssignmentFingerprint([2; 32]),
        PrecisionPolicy::Exact,
    );
    let evidence = NumericalEvidence {
        key: key.clone(),
        assessment: NumericalAssessment::exact(),
    };
    assert!(evidence.qualifies(&key));
    // Any identity change invalidates the record.
    let mut other = key.clone();
    other.assignment = AssignmentFingerprint([9; 32]);
    assert!(!evidence.qualifies(&other));
    let mut other = key.clone();
    other.target = target("metal");
    assert!(!evidence.qualifies(&other));
}

// ---------------------------------------------------------------------------
// Reduction strategies
// ---------------------------------------------------------------------------

fn sum_reduction(order: ReductionOrder) -> ReductionNode {
    ReductionNode {
        operand: GraphValueId(0),
        axis: 1,
        op: ReduceOp::Sum,
        order,
        accumulator: DType::F32,
        result: ValueType::Scalar(DType::F32),
    }
}

#[test]
fn universal_reduction_strategy_is_exact_and_ordered() {
    let operand = TensorType::new(
        vec![ExtentExpr::Static(4), ExtentExpr::Static(8)],
        Elem::Dtype(DType::F16),
    );
    let strategy =
        reduction::universal(&sum_reduction(ReductionOrder::Ascending), &operand).expect("legal");
    assert_eq!(strategy.kind, ReductionStrategyKind::ParallelOuter);
    assert_eq!(strategy.reassociation, ReassociationAdmission::Ordered);
    assert_eq!(strategy.numerical, NumericalTransfer::Exact);
    assert_eq!(strategy.tie_rule, TieRule::SmallerCoordinateIndex);
    assert_eq!(strategy.identity, ReductionIdentity::Zero);
    // Floating f16 input accumulates and results in f32 (registry rule).
    assert_eq!(strategy.accumulator, DType::F32);
    assert_eq!(strategy.resources, ReductionResources::UNIVERSAL);
    // Parallel outer coordinates; one participant per output; serial
    // ascending reduced axis; one publication.
    match &strategy.topology {
        ReductionTopology::ParallelOuter { outer_axes, inner } => {
            assert_eq!(outer_axes, &[ExtentExpr::Static(4)]);
            match inner.as_ref() {
                ReductionTopology::SerialAxis { axis, length } => {
                    assert_eq!(*axis, 1);
                    assert_eq!(*length, ExtentExpr::Static(8));
                }
                other => panic!("expected a serial inner fold, got {other:?}"),
            }
        }
        other => panic!("expected parallel-outer topology, got {other:?}"),
    }
}

#[test]
fn argmax_preserves_ties_and_requires_nonempty() {
    let operand = TensorType::new(vec![ExtentExpr::Static(4)], Elem::Dtype(DType::F16));
    let reduction = ReductionNode {
        operand: GraphValueId(0),
        axis: 0,
        op: ReduceOp::Argmax,
        order: ReductionOrder::Ascending,
        accumulator: DType::I32,
        result: ValueType::Scalar(DType::I32),
    };
    let strategy = reduction::universal(&reduction, &operand).expect("legal");
    assert_eq!(strategy.identity, ReductionIdentity::FirstElementNonEmpty);
    assert_eq!(strategy.accumulator, DType::I32);
    assert_eq!(
        strategy.preconditions,
        vec![ReductionPrecondition::NonEmpty {
            axis: 0,
            length: ExtentExpr::Static(4)
        }]
    );
    // Argmax never reassociates.
    assert_eq!(
        reduction::reassociable(
            &reduction,
            &operand,
            ReductionTopology::Tree {
                fan_in: 2,
                depth: 2,
                inner: Box::new(ReductionTopology::SerialAxis {
                    axis: 0,
                    length: ExtentExpr::Static(4),
                }),
            },
            ReassociationAdmission::SourceUnordered,
        ),
        Err(ReductionAdmissionError::ArgmaxNeverReassociates)
    );
}

#[test]
fn reassociation_requires_admission() {
    let operand = TensorType::new(
        vec![ExtentExpr::Static(4), ExtentExpr::Static(8)],
        Elem::Dtype(DType::F32),
    );
    let topology = ReductionTopology::Tree {
        fan_in: 2,
        depth: 3,
        inner: Box::new(ReductionTopology::SerialAxis {
            axis: 1,
            length: ExtentExpr::Static(8),
        }),
    };
    // The source ordered the reduction: source-level admission is refused.
    assert_eq!(
        reduction::reassociable(
            &sum_reduction(ReductionOrder::Ascending),
            &operand,
            topology.clone(),
            ReassociationAdmission::SourceUnordered,
        ),
        Err(ReductionAdmissionError::AscendingOrder)
    );
    // The source marked it unordered: admitted with a reassociate transfer.
    let strategy = reduction::reassociable(
        &sum_reduction(ReductionOrder::Unordered),
        &operand,
        topology.clone(),
        ReassociationAdmission::SourceUnordered,
    )
    .expect("admitted");
    assert_eq!(strategy.kind, ReductionStrategyKind::Tree);
    assert_eq!(
        strategy.numerical,
        NumericalTransfer::Reassociate {
            op: ReduceOp::Sum,
            topology
        }
    );
    // Caller policy/evidence can admit even an ordered source reduction.
    assert!(reduction::reassociable(
        &sum_reduction(ReductionOrder::Ascending),
        &operand,
        ReductionTopology::Subgroup {
            width: 32,
            inner: Box::new(ReductionTopology::SerialAxis {
                axis: 1,
                length: ExtentExpr::Static(8),
            }),
        },
        ReassociationAdmission::PolicyOrEvidence,
    )
    .is_ok());
}

// ---------------------------------------------------------------------------
// Universal arbitrary-rank iteration
// ---------------------------------------------------------------------------

#[test]
fn linear_iteration_maps_use_checked_totals() {
    let empty: BTreeMap<RuntimeExtentId, RuntimeExtent> = BTreeMap::new();
    let map = LinearIterationMap::linear(&[ExtentExpr::Static(4), ExtentExpr::Static(8)], &empty)
        .expect("static extents map");
    assert_eq!(map.total, LinearTotal::Static(32));
    assert_eq!(map.delinearize(9), Some(vec![1, 1]));
    assert_eq!(map.delinearize(32), None);
    assert_eq!(map.launch_condition(), LaunchCondition::Execute);
    // Zero work is a retained launch condition, never a submitted grid.
    let zero = LinearIterationMap::linear(&[ExtentExpr::Static(0)], &empty).expect("maps");
    assert_eq!(zero.launch_condition(), LaunchCondition::AlwaysSkip);
    // Static overflow is infeasible, never wrapped.
    assert_eq!(
        LinearIterationMap::linear(
            &[ExtentExpr::Static(u64::MAX), ExtentExpr::Static(2)],
            &empty
        ),
        Err(LinearMapError::StaticOverflow)
    );
    // Runtime extents retain the exact runtime product with a checked
    // capacity bound.
    let mut extents = BTreeMap::new();
    extents.insert(
        RuntimeExtentId(0),
        RuntimeExtent {
            id: RuntimeExtentId(0),
            value: seismic_lang::logical::RuntimeScalarExpr::Const(3),
            capacity: 4,
            expected: None,
        },
    );
    let map = LinearIterationMap::linear(
        &[
            ExtentExpr::Runtime(RuntimeExtentId(0)),
            ExtentExpr::Static(4),
        ],
        &extents,
    )
    .expect("runtime extents map");
    match &map.total {
        LinearTotal::Runtime {
            product, capacity, ..
        } => {
            assert_eq!(*capacity, 16);
            // Canonical retained product: no leading unit factor.
            assert_eq!(
                *product,
                seismic_lang::logical::RuntimeScalarExpr::Mul(
                    Box::new(seismic_lang::logical::RuntimeScalarExpr::Extent(
                        RuntimeExtentId(0)
                    )),
                    Box::new(seismic_lang::logical::RuntimeScalarExpr::Const(4)),
                )
            );
        }
        other => panic!("expected a retained runtime total, got {other:?}"),
    }
    // The universal atomic-add strategy serializes the domain.
    let serialized = LinearIterationMap::serialized(&map);
    assert!(serialized.serialized);
    assert_eq!(serialized.traversal, Traversal::OnePass);
    assert!(!serialized.tail_mask);
}

// ---------------------------------------------------------------------------
// Formation over constructed logical programs
// ---------------------------------------------------------------------------

#[test]
fn formation_routes_nodes_and_discharges_obligations() {
    let logical = entry_graph(
        &[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32, i: index[N]) -> f32:\n    return reduce(f32(x), 0, sum) + 1.0 / x[i]\n",
        )],
        "sum",
        "cpu",
        &[("N", 16)],
    );
    let graph = logical.graph(
        logical
            .choice(logical.entry_choice)
            .alternatives
            .iter()
            .next()
            .unwrap()
            .graph,
    );
    let facts = GraphFacts::collect(&graph, &logical.runtime_extents);

    let mut reductions = 0;
    let mut proved_reads = 0;
    let mut checked_divisions = 0;
    for id in graph.root.nodes.ids() {
        let node = &graph.root.nodes[id];
        match universal_node(id, node, &facts).expect("every node forms") {
            UniversalNode::Reduction(reduction) => {
                reductions += 1;
                // The reduced axis is statically nonempty: proved.
                assert!(reduction.preconditions.iter().all(|(_, discharge)| {
                    matches!(discharge, ObligationDischarge::StaticallyProved(_))
                }));
                assert_eq!(reduction.strategy.numerical, NumericalTransfer::Exact);
            }
            UniversalNode::Primitive(formed) => {
                if matches!(
                    formed.form,
                    UniversalForm::CheckedAccess {
                        access: DataAccess::Load
                    }
                ) {
                    // The read uses the index parameter, refined to N: proved.
                    assert!(formed.obligations.iter().all(|(_, discharge)| {
                        matches!(discharge, ObligationDischarge::StaticallyProved(_))
                    }));
                    proved_reads += 1;
                }
                if matches!(
                    formed.form,
                    UniversalForm::FloatArithmetic { dtype: DType::F32 }
                ) {
                    // The division's divisor is a runtime value: planned check
                    // with inactive behavior and a status write.
                    for (obligation, discharge) in &formed.obligations {
                        if matches!(obligation, SafetyObligation::DivisorNonZero { .. }) {
                            match discharge {
                                ObligationDischarge::RuntimeChecked(check) => {
                                    assert_eq!(check.status.kind, SafetyKind::DivisionByZero);
                                    assert_eq!(check.inactive, InactiveBehavior::SkipOperation);
                                    checked_divisions += 1;
                                }
                                other => panic!("expected a checked divisor, got {other:?}"),
                            }
                        }
                    }
                }
                assert_eq!(formed.numerical, NumericalTransfer::Exact);
            }
            other => panic!("unexpected universal node {other:?}"),
        }
    }
    assert_eq!(reductions, 1);
    assert!(proved_reads >= 1);
    assert_eq!(checked_divisions, 1);
}

#[test]
fn formation_routes_calls_and_loops() {
    let logical = entry_graph(
        &[(
            "kernel.seismic",
            "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n",
        )],
        "linear",
        "cpu",
        &[("M", 4), ("N", 8)],
    );
    // The entry graph is one call: routed to `invoke`, never formed.
    let entry = logical.graph(
        logical
            .choice(logical.entry_choice)
            .alternatives
            .iter()
            .next()
            .unwrap()
            .graph,
    );
    let facts = GraphFacts::collect(&entry, &logical.runtime_extents);
    for id in entry.root.nodes.ids() {
        match universal_node(id, &entry.root.nodes[id], &facts).expect("forms") {
            UniversalNode::Call(call) => {
                assert!(call.synchronous);
                assert_eq!(call.choice, ChoiceId(1));
                // Two shared borrows and one move become boundary leaves.
                assert_eq!(call.inputs.len(), 3);
                assert!(!call.inputs.is_empty() && !call.results.is_empty());
            }
            UniversalNode::Primitive(formed) => {
                // Only primitive data flow remains in the entry body.
                assert!(!matches!(formed.form, UniversalForm::StorageAllocation));
            }
            other => panic!("unexpected node {other:?} in the entry body"),
        }
    }

    // The callee graph: an independent outer loop with a disjoint-write join
    // and an ordered inner loop with carries.
    let add_choice = logical
        .choices
        .iter()
        .find(|choice| choice.interface.name == "add")
        .expect("the call created its own choice");
    let graph = logical.graph(add_choice.alternatives.iter().next().unwrap().graph);
    let facts = GraphFacts::collect(&graph, &logical.runtime_extents);
    let mut independent = 0;
    let mut ordered = 0;
    let mut writes = 0;
    // Loops and their bodies nest: walk regions recursively.
    fn scan(
        region: &seismic_lang::logical::GraphRegion,
        facts: &GraphFacts,
        independent: &mut usize,
        ordered: &mut usize,
        writes: &mut usize,
    ) {
        for id in region.nodes.ids() {
            let node = &region.nodes[id];
            match universal_node(id, node, facts).expect("forms") {
                UniversalNode::Loop(loop_node) => {
                    assert!(!loop_node.serialized_for_atomic);
                    match loop_node.kind {
                        seismic_lang::sir::LoopKind::Independent => {
                            *independent += 1;
                            // The independent domain carries the universal
                            // linear map over the runtime-bounded axis.
                            let map = loop_node.iteration.as_ref().expect("mapped");
                            assert!(!map.serialized);
                            assert!(map.launch_condition() == LaunchCondition::Execute);
                        }
                        seismic_lang::sir::LoopKind::Ordered => {
                            *ordered += 1;
                            assert!(loop_node.iteration.is_none());
                            assert!(!loop_node.carries.is_empty());
                        }
                    }
                }
                UniversalNode::Primitive(formed) => {
                    if matches!(
                        formed.form,
                        UniversalForm::CheckedAccess {
                            access: DataAccess::Store
                        }
                    ) {
                        *writes += 1;
                        // Writes through loop binders are index-refined and
                        // statically proved.
                        assert!(formed.obligations.iter().all(|(_, discharge)| {
                            matches!(discharge, ObligationDischarge::StaticallyProved(_))
                        }));
                    }
                }
                other => panic!("unexpected node {other:?} in the callee body"),
            }
            // Recurse into nested regions.
            match &node.kind {
                LogicalNodeKind::Loop(loop_node) => {
                    scan(&loop_node.body, facts, independent, ordered, writes)
                }
                LogicalNodeKind::If(if_node) => {
                    scan(&if_node.then_region, facts, independent, ordered, writes);
                    scan(&if_node.else_region, facts, independent, ordered, writes);
                }
                _ => {}
            }
        }
    }
    scan(
        &graph.root,
        &facts,
        &mut independent,
        &mut ordered,
        &mut writes,
    );
    assert!(independent >= 1 && ordered >= 1 && writes >= 1);
}

#[test]
fn universal_atomic_add_serializes_the_independent_domain() {
    let logical = entry_graph(
        &[(
            "atomic.seismic",
            "fn hist[N](x: &tensor[N] f32, out: tensor[64] f32) -> f32 for metal:\n    let mut acc = out\n    parallel for i in 0..N:\n        atomic(add, acc[0], x[i])\n    return reduce(f32(acc), 0, sum)\n",
        )],
        "hist",
        "metal",
        &[("N", 128)],
    );
    let graph = logical.graph(
        logical
            .choice(logical.entry_choice)
            .alternatives
            .iter()
            .next()
            .unwrap()
            .graph,
    );
    let facts = GraphFacts::collect(&graph, &logical.runtime_extents);
    let mut serialized_loops = 0;
    for id in graph.root.nodes.ids() {
        let node = &graph.root.nodes[id];
        if let UniversalNode::Loop(loop_node) = universal_node(id, node, &facts).expect("forms") {
            if loop_node.serialized_for_atomic {
                serialized_loops += 1;
                let map = loop_node.iteration.as_ref().expect("mapped");
                assert!(map.serialized);
                assert_eq!(map.traversal, Traversal::OnePass);
                // The admitted atomic operation is registry-legal (f32).
                assert!(loop_node.joins.iter().any(|(_, join)| {
                    matches!(join, StateJoin::Atomic { operations } if operations.iter().all(
                        |operation| matches!(operation,
                            seismic_lang::logical::AtomicOperation { dtype, .. }
                                if seismic_lang::intrinsics::atomic_dtype(*dtype))
                    ))
                }));
            }
        }
    }
    assert_eq!(serialized_loops, 1);
}

// ---------------------------------------------------------------------------
// Formation → builder wiring against a local sealed dialect
// ---------------------------------------------------------------------------

#[test]
fn formation_wires_into_the_family_builder() {
    use seismic_realization::executable::{
        EffectiveTargetProfile, ExecutableDialect, InvariantReport, Legalized, NodeRef,
        ObligationRef, PhysicalConsequences, PhysicalPrimitive, PlanFamilyBuilder, PlanValues,
        TargetLimits,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestOp {
        Compute,
        Access,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestDialect;
    impl seismic_realization::executable::sealed::Sealed for TestDialect {}

    impl ExecutableDialect for TestDialect {
        type Op = TestOp;
        type LayoutTemplate = u64;
        type ResolvedLayout = u64;

        fn legalize(p: &PhysicalPrimitive, _t: &EffectiveTargetProfile) -> Legalized<TestOp> {
            // Capabilities have no portable opcode: the optional-capability
            // route. Every registry primitive legalizes.
            if matches!(p.op, seismic_lang::logical::PrimitiveOp::Capability(_)) {
                return Legalized::Inapplicable {
                    reason: "no portable opcode for a capability".into(),
                };
            }
            let op = match &p.op {
                seismic_lang::logical::PrimitiveOp::Primitive(id)
                    if matches!(
                        id,
                        PrimitiveId::ElementRead { .. }
                            | PrimitiveId::ElementWrite { .. }
                            | PrimitiveId::PackedRead(_)
                    ) =>
                {
                    TestOp::Access
                }
                _ => TestOp::Compute,
            };
            Legalized::Ops(NonEmpty::new(vec![op]).expect("one opcode"))
        }

        fn consequences(_op: &TestOp) -> PhysicalConsequences {
            // The universal consequences constructor provides the exact
            // hard-resource and native-contract facts.
            universal_consequences(0, 0)
        }

        fn public_layout(tensor: &TensorType) -> u64 {
            match &tensor.elem {
                Elem::Dtype(dtype) => u64::from(dtype.bytes()),
                _ => 4,
            }
        }

        fn internal_layout(tensor: &TensorType) -> u64 {
            Self::public_layout(tensor)
        }

        fn resolve_layout(layout: &u64, _values: &PlanValues) -> Result<u64, InvariantReport> {
            Ok(*layout)
        }
    }

    let profile = EffectiveTargetProfile {
        backend: "cpu".into(),
        capability_fingerprint: "test-fingerprint".into(),
        toolchain_fingerprint: "test-toolchain".into(),
        effective_signatures: BTreeSet::new(),
        limits: TargetLimits {
            max_participants: 1024,
            max_workgroups_axis: [1024, 1, 1],
            max_workgroup_bytes: 32768,
            max_explicit_private_bytes: 32768,
            max_direct_bindings: 32,
            max_argument_table_bytes: 4096,
            max_device_bytes: 1 << 30,
        },
    };

    let logical = entry_graph(
        &[(
            "reduce.seismic",
            "fn sum[N](x: &tensor[N] f32, i: index[N]) -> f32:\n    return reduce(f32(x), 0, sum) + 1.0 / x[i]\n",
        )],
        "sum",
        "cpu",
        &[("N", 16)],
    );
    let family =
        PlanFamilyBuilder::<TestDialect>::from_logical(&logical).expect("the family opens");
    let mut builder = family
        .alternative(logical.entry_choice, 0)
        .expect("the alternative opens");
    let graph = logical.graph(
        logical
            .choice(logical.entry_choice)
            .alternatives
            .iter()
            .next()
            .unwrap()
            .graph,
    );
    let facts = GraphFacts::collect(&graph, &logical.runtime_extents);

    let mut mapped = 0;
    let mut proved = 0;
    let mut runtime_checked = 0;
    for id in graph.root.nodes.ids() {
        let node = &graph.root.nodes[id];
        let LogicalNodeKind::Primitive(_) = &node.kind else {
            continue;
        };
        let formed = form_primitive(id, node, &facts).expect("the primitive forms");
        // The exact physical primitive carries the real canonical types.
        let physical = formed.physical_primitive();
        assert_eq!(physical.inputs.len(), node.inputs.len());
        assert_eq!(physical.results.len(), node.outputs.len());
        assert_eq!(
            physical.results,
            node.outputs
                .iter()
                .map(|o| o.ty.clone())
                .collect::<Vec<_>>()
        );
        legalize_and_map_primitive(
            &mut builder,
            &profile,
            NodeRef {
                region: Vec::new(),
                node: id,
            },
            &formed,
        )
        .expect("the primitive maps");
        mapped += 1;
        for (index, (_, classified)) in formed.obligations.iter().enumerate() {
            let reference = ObligationRef {
                node: NodeRef {
                    region: Vec::new(),
                    node: id,
                },
                index,
            };
            discharge_with_builder(&mut builder, reference, classified, |_| {
                Legalized::Ops(NonEmpty::new(vec![TestOp::Compute]).expect("one opcode"))
            })
            .expect("the obligation discharges");
            match classified {
                ObligationDischarge::StaticallyProved { .. } => proved += 1,
                ObligationDischarge::RuntimeChecked { .. } => runtime_checked += 1,
                ObligationDischarge::StaticallyImpossible { .. } => {}
            }
        }
    }
    assert!(mapped >= 3, "the elementwise/cast/read nodes all map");
    assert!(proved >= 1, "the index-refined read is proved");
    assert!(
        runtime_checked >= 1,
        "the division divisor is runtime checked"
    );
}

#[test]
fn capability_applications_route_to_the_capability_path() {
    let logical = entry_graph(
        &[(
            "cap.seismic",
            "fn f[M](x: &tensor[M] f32) -> f32 for metal requires metal.subgroup:\n    return metal.subgroup.simd_sum(x[0])\n",
        )],
        "f",
        "metal",
        &[("M", 4)],
    );
    let graph = logical.graph(
        logical
            .choice(logical.entry_choice)
            .alternatives
            .iter()
            .next()
            .unwrap()
            .graph,
    );
    let facts = GraphFacts::collect(&graph, &logical.runtime_extents);
    let mut capability_routes = 0;
    let mut plain_primitives = 0;
    for id in graph.root.nodes.ids() {
        let node = &graph.root.nodes[id];
        let LogicalNodeKind::Primitive(_) = &node.kind else {
            continue;
        };
        match form_primitive(id, node, &facts) {
            Err(FormationError::RequiresCapability { intrinsic }) => {
                capability_routes += 1;
                assert_eq!(intrinsic.path(), "metal.subgroup.simd_sum");
            }
            Ok(_) => plain_primitives += 1,
            other => panic!("unexpected formation failure: {other:?}"),
        }
    }
    assert!(capability_routes >= 1);
    assert!(plain_primitives >= 1);
}
