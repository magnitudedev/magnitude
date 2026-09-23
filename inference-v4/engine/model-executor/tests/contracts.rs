//! Pure launch-control validation retained after the executor/domain rewrite.
use magnitude_model_executor::{
    batching::Demand, FeatureRef, Operation, RequestId, ResourceDomainId, Shaping, TokenId,
    WorkKind,
};
use std::sync::Arc;

fn feature(id: u64) -> FeatureRef {
    FeatureRef::logical(
        ResourceDomainId::new(format!("contract-{id}")).unwrap(),
        4,
        4,
    )
    .unwrap()
}

#[test]
fn shaping_validation_owns_the_executor_wire_contract() {
    let identity = Shaping::default();
    assert_eq!(identity.validate(), Ok(identity));
    assert!(identity.is_identity());
    assert!(!identity.uses_history());

    for invalid in [
        Shaping {
            temperature: -1.0,
            ..identity
        },
        Shaping {
            top_p: 0.0,
            ..identity
        },
        Shaping {
            min_p: 1.1,
            ..identity
        },
        Shaping {
            repetition_penalty: 0.0,
            ..identity
        },
        Shaping {
            presence_penalty: f32::NAN,
            ..identity
        },
        Shaping {
            top_k: u32::MAX,
            ..identity
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn operation_validation_checks_the_complete_selection_row_contract() {
    let selection =
        |shaping: Shaping, history: Option<Arc<[i32]>>| magnitude_model_executor::SelectSpec {
            sampling: magnitude_model_executor::Sampling::Categorical,
            seed: 7,
            position: 3,
            domain: 1,
            mask: None,
            shaping,
            history,
        };
    let selected = |select| Operation::Forward {
        request: RequestId(1),
        kind: WorkKind::Decode,
        tokens: vec![TokenId(2)],
        position: 3,
        conditioning: None,
        demand: Demand::SELECT,
        select: vec![select],
        committed: 1,
    };

    assert!(selected(selection(Shaping::default(), None))
        .validate()
        .is_ok());
    let penalties = Shaping {
        repetition_penalty: 1.1,
        ..Shaping::default()
    };
    assert!(selected(selection(penalties, None)).validate().is_err());
    assert!(selected(selection(penalties, Some(vec![-1; 63].into())))
        .validate()
        .is_err());
    assert!(selected(selection(penalties, Some(vec![-1; 64].into())))
        .validate()
        .is_ok());

    let per_row = (0..3)
        .map(|position| {
            let mut spec = selection(Shaping::default(), None);
            spec.position = position;
            spec
        })
        .collect::<Vec<_>>();
    let verify = Operation::Forward {
        request: RequestId(2),
        kind: WorkKind::Verify,
        tokens: vec![TokenId(4), TokenId(5), TokenId(6)],
        position: 7,
        conditioning: None,
        demand: Demand::SELECT,
        select: per_row.clone(),
        committed: 1,
    };
    assert!(verify.validate().is_ok());
    assert_eq!(verify.selection_for_row(2), per_row.get(2));

    let mut missing_verify_row = verify.clone();
    let Operation::Forward { select, .. } = &mut missing_verify_row else {
        unreachable!()
    };
    select.pop();
    assert!(matches!(
        missing_verify_row.validate(),
        Err(magnitude_model_executor::OperationError::SelectionRows {
            kind: WorkKind::Verify,
            expected: 3,
            actual: 2,
        })
    ));

    let finishing_prefill = Operation::Forward {
        request: RequestId(3),
        kind: WorkKind::Prefill,
        tokens: vec![TokenId(7), TokenId(8), TokenId(9)],
        position: 0,
        conditioning: None,
        demand: Demand::SELECT,
        select: vec![selection(Shaping::default(), None)],
        committed: 3,
    };
    assert!(finishing_prefill.validate().is_ok());
    assert!(finishing_prefill.selection_for_row(0).is_none());
    assert!(finishing_prefill.selection_for_row(1).is_none());
    assert!(finishing_prefill.selection_for_row(2).is_some());

    let invalid = Operation::Forward {
        request: RequestId(4),
        kind: WorkKind::Replay,
        tokens: vec![TokenId(1)],
        position: 0,
        conditioning: None,
        demand: Demand::SELECT,
        select: vec![selection(Shaping::default(), None)],
        committed: 1,
    };
    assert!(matches!(
        invalid.validate(),
        Err(magnitude_model_executor::OperationError::SelectionRows {
            expected: 0,
            actual: 1,
            ..
        })
    ));
}

#[test]
fn head_feature_span_must_match_rows_and_have_a_checked_nonempty_range() {
    let mismatched = Operation::Head {
        request: RequestId(3),
        tokens: vec![TokenId(4), TokenId(5)],
        conditioning: magnitude_model_executor::FeatureSpan {
            features: feature(5),
            start: 0,
            count: 1,
        },
        position: 0,
        demand: Demand::NONE,
    };
    assert!(matches!(
        mismatched.validate(),
        Err(magnitude_model_executor::OperationError::FeatureSpan { count: 1, rows: 2 })
    ));

    let overflow = Operation::Head {
        request: RequestId(3),
        tokens: vec![TokenId(4)],
        conditioning: magnitude_model_executor::FeatureSpan {
            features: feature(5),
            start: usize::MAX,
            count: 1,
        },
        position: 0,
        demand: Demand::NONE,
    };
    assert!(matches!(
        overflow.validate(),
        Err(magnitude_model_executor::OperationError::FeatureSpan { count: 1, rows: 1 })
    ));
}
