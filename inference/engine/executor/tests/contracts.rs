//! Pure launch-control validation retained after the executor/domain rewrite.
use magnitude_executor::{
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
        |shaping: Shaping, history: Option<Arc<[i32]>>| magnitude_executor::SelectSpec {
            sampling: magnitude_executor::Sampling::Categorical,
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
        prime: None,
        front: Vec::new(),
        chain: None,
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
        prime: None,
        front: Vec::new(),
        chain: None,
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
        Err(magnitude_executor::OperationError::SelectionRows {
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
        prime: None,
        front: Vec::new(),
        chain: None,
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
        prime: None,
        front: Vec::new(),
        chain: None,
    };
    assert!(matches!(
        invalid.validate(),
        Err(magnitude_executor::OperationError::SelectionRows {
            expected: 0,
            actual: 1,
            ..
        })
    ));
}

#[test]
fn head_conditioning_rows_must_match_its_entry_rows() {
    let rows = |count: usize| {
        magnitude_executor::FeatureRows::new(vec![0u8; 4 * count].into(), count).unwrap()
    };
    let head = |tokens: usize, conditioning: usize, proposals: usize| Operation::Head {
        request: RequestId(3),
        phase: magnitude_executor::HeadPhase::Generation,
        tokens: vec![TokenId(4); tokens],
        conditioning: rows(conditioning),
        position: 0,
        proposals: vec![select(); proposals],
        form: magnitude_executor::DraftForm::Chained,
    };
    assert!(matches!(
        head(2, 1, 0).validate(),
        Err(magnitude_executor::OperationError::FeatureSpan { count: 1, rows: 2 })
    ));
    let drafting = head(2, 2, 3);
    assert!(drafting.validate().is_ok());
    // Entry rows plus one chained row per proposal after the first.
    assert_eq!(drafting.row_count(), 4);
    assert_eq!(drafting.demand(), Demand::SELECT);
    assert_eq!(head(1, 1, 0).demand(), Demand::NONE);
    assert!(magnitude_executor::FeatureRows::new(vec![0u8; 5].into(), 2).is_err());
    assert!(rows(2)
        .concat(&rows(1))
        .is_ok_and(|joined| joined.rows() == 3));
    assert!(rows(3).slice(1, 3).is_err());
}

/// Front rows run on a chain, so they trail a chained decode pass's one
/// exact row, within a chained launch's rows: a pass of one and of every
/// row past the exact one is valid; front rows of no chain, alone, behind a
/// verification's proposals, behind a decode of rows a decision may accept
/// beyond its exact row, or past a chained launch's rows are not, and
/// neither are acceptable rows with no exact row before them.
#[test]
fn operation_validation_admits_only_a_chained_decodes_front_rows() {
    use magnitude_executor::{ChainEntry, OperationError};
    let pass = |kind: WorkKind, tokens: &[u32], selects: usize, committed: usize, fronts: usize, chain: bool| Operation::Forward {
        request: RequestId(1),
        kind,
        tokens: tokens.iter().copied().map(TokenId).collect(),
        position: 3,
        conditioning: None,
        demand: Demand::SELECT | Demand::FEATURES,
        select: (0..selects).map(|_| select()).collect(),
        committed,
        prime: None,
        front: (0..fronts)
            .map(|_| magnitude_executor::FrontRow { select: select(), kind: magnitude_executor::FrontKind::Host(magnitude_executor::GuessKind::Chained) })
            .collect(),
        chain: chain.then_some(ChainEntry { position: 2 }),
    };
    let rows = magnitude_executor::CHAINED_ROWS as u32;
    let tokens = |count: u32| (2..2 + count).collect::<Vec<_>>();
    assert!(pass(WorkKind::Decode, &[2, 3], 1, 1, 1, true).validate().is_ok(), "a pass of one front row");
    assert!(pass(WorkKind::Decode, &tokens(rows), 1, 1, rows as usize - 1, true).validate().is_ok(), "a pass of a chained launch's rows");
    assert!(
        matches!(pass(WorkKind::Decode, &[2, 3], 1, 1, 1, false).validate(), Err(OperationError::FrontRows)),
        "a front row of no chain"
    );
    assert!(
        matches!(pass(WorkKind::Decode, &tokens(rows + 1), 1, 1, rows as usize, true).validate(), Err(OperationError::FrontRows)),
        "front rows past a chained launch's rows"
    );
    assert!(pass(WorkKind::Decode, &[2], 0, 0, 1, true).validate().is_err(), "a front row alone");
    assert!(pass(WorkKind::Verify, &[2, 3, 4], 2, 1, 1, true).validate().is_err(), "a front row behind a verification's proposals");
    assert!(pass(WorkKind::Decode, &[2, 3, 4], 1, 1, 1, true).validate().is_err(), "a front row behind two rows");
    assert!(pass(WorkKind::Verify, &[2, 3], 2, 0, 0, false).validate().is_err(), "acceptable rows with no exact row");
}

/// A chain entry follows a selecting decode pass of one exact row, front
/// rows or none, or the prompt's closing chunk priming across its boundary,
/// each demanding the features its head entry reads; a verification, a
/// chunk that does not prime across its boundary, or a forward demanding no
/// features takes none.
#[test]
fn operation_validation_admits_a_chain_behind_a_decode_or_the_closing_chunk() {
    use magnitude_executor::{ChainEntry, OperationError, Priming};
    let chained = |kind: WorkKind, tokens: &[u32], demand: Demand, selects: usize, committed: usize, prime: Option<Priming>, fronts: usize| Operation::Forward {
        request: RequestId(1),
        kind,
        tokens: tokens.iter().copied().map(TokenId).collect(),
        position: 3,
        conditioning: None,
        demand,
        select: (0..selects).map(|_| select()).collect(),
        committed,
        prime,
        front: (0..fronts)
            .map(|_| magnitude_executor::FrontRow { select: select(), kind: magnitude_executor::FrontKind::Device })
            .collect(),
        chain: Some(ChainEntry { position: 2 }),
    };
    let entered = Demand::SELECT | Demand::FEATURES;
    let boundary = |boundary| Some(Priming { boundary, tokens: vec![TokenId(2), TokenId(3)], position: 2, draft_from: 2 });
    assert!(chained(WorkKind::Decode, &[2], entered, 1, 1, None, 0).validate().is_ok(), "a one-row decode");
    assert!(chained(WorkKind::Decode, &[2, 3, 4], entered, 1, 1, None, 2).validate().is_ok(), "a front pass");
    assert!(chained(WorkKind::Prefill, &[2, 3], entered, 1, 2, boundary(true), 0).validate().is_ok(), "the closing chunk");
    for (operation, case) in [
        (chained(WorkKind::Decode, &[2], Demand::SELECT, 1, 1, None, 0), "a decode demanding no features"),
        (chained(WorkKind::Verify, &[2, 3], entered, 2, 1, None, 0), "a verification"),
        (chained(WorkKind::Prefill, &[2, 3], entered, 1, 2, None, 0), "a chunk priming nothing"),
        (chained(WorkKind::Prefill, &[2, 3], entered, 1, 2, boundary(false), 0), "a chunk priming within itself"),
    ] {
        assert!(matches!(operation.validate(), Err(OperationError::Chain)), "{case}");
    }
}
