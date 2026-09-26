//! Pure launch-control validation retained after the executor/domain rewrite.
use magnitude_model_executor::{
    batching::Demand, Operation, RequestId, Sampling, SelectSpec, Shaping, TokenId, WorkKind,
};
use std::sync::Arc;

fn select() -> SelectSpec {
    SelectSpec {
        sampling: Sampling::Greedy,
        seed: 0,
        position: 0,
        domain: 0,
        mask: None,
        shaping: Shaping::default(),
        history: None,
    }
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
fn head_conditioning_rows_must_match_its_entry_rows() {
    let rows = |count: usize| {
        magnitude_model_executor::FeatureRows::new(vec![0u8; 4 * count].into(), count).unwrap()
    };
    let head = |tokens: usize, conditioning: usize, proposals: usize| Operation::Head {
        request: RequestId(3),
        tokens: vec![TokenId(4); tokens],
        conditioning: rows(conditioning),
        position: 0,
        proposals: vec![select(); proposals],
    };
    assert!(matches!(
        head(2, 1, 0).validate(),
        Err(magnitude_model_executor::OperationError::FeatureSpan { count: 1, rows: 2 })
    ));
    let drafting = head(2, 2, 3);
    assert!(drafting.validate().is_ok());
    // Entry rows plus one chained row per proposal after the first.
    assert_eq!(drafting.row_count(), 4);
    assert_eq!(drafting.demand(), Demand::SELECT);
    assert_eq!(head(1, 1, 0).demand(), Demand::NONE);
    assert!(magnitude_model_executor::FeatureRows::new(vec![0u8; 5].into(), 2).is_err());
    assert!(rows(2)
        .concat(&rows(1))
        .is_ok_and(|joined| joined.rows() == 3));
    assert!(rows(3).slice(1, 3).is_err());
}
