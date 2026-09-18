use seismic_engine::{
    inputs::TokenId,
    models::sequence::OwnedSequence,
    state::{ComponentSpec, SequenceState, StateStore},
};
use seismic_lang::types::DType;
use seismic_runtime::Device;
use std::rc::Rc;

fn store() -> Rc<StateStore> {
    StateStore::new(
        Rc::new(Device::cpu()),
        16,
        32,
        vec![16],
        vec![ComponentSpec {
            shape: vec![4],
            dtype: DType::F32,
        }],
    )
    .unwrap()
}
fn execute(
    state: &mut SequenceState,
    count: usize,
    value: u8,
) -> Result<Option<TokenId>, seismic_runtime::Error> {
    let mut advance = state.begin(count)?;
    advance.execute(|b| {
        b.following[0].write(&[value; 16])?;
        Ok(())
    })?;
    advance.commit()?;
    Ok(Some(TokenId(7)))
}
fn values(sequence: &OwnedSequence) -> [u8; 16] {
    let state = sequence.checkpoint().unwrap().fork();
    let mut bytes = [0; 16];
    state.values()[0].read(&mut bytes).unwrap();
    bytes
}
#[test]
fn completed_work_is_invisible_until_acceptance_and_rejection_releases_claims() {
    let store = store();
    let sequence = OwnedSequence::new(store.create().unwrap());
    let before = sequence.checkpoint().unwrap();
    let row = sequence
        .prepare_completed(0, 3, |s| execute(s, 3, 0x22))
        .unwrap();
    assert_eq!(sequence.position(), 0);
    assert!(sequence.pending());
    assert!(sequence.checkpoint().is_err());
    assert!(sequence
        .prepare_completed(0, 1, |s| execute(s, 1, 0x11))
        .is_err());
    assert_eq!(store.occupied_rows(), 3);
    drop(row);
    assert!(!sequence.pending());
    assert_eq!(store.occupied_rows(), 0);
    assert_eq!(values(&sequence), [0; 16]);
    let mut row = sequence
        .prepare_completed(0, 3, |s| execute(s, 3, 0x33))
        .unwrap();
    assert!(row.is_complete());
    assert_eq!(row.selected().unwrap(), Some(TokenId(7)));
    row.commit().unwrap();
    assert!(row.commit().is_err());
    assert_eq!(sequence.position(), 3);
    assert_eq!(values(&sequence), [0x33; 16]);
    assert_eq!(before.position(), 0);
    assert_eq!(values(&OwnedSequence::new(before.fork())), [0; 16]);
    assert!(sequence
        .prepare_completed(0, 1, |s| execute(s, 1, 0))
        .is_err());
}
#[test]
fn preparation_failure_wrong_extent_and_unwind_leave_owner_reusable() {
    let store = store();
    let sequence = OwnedSequence::new(store.create().unwrap());
    assert!(sequence
        .prepare_completed(0, 2, |s| {
            execute(s, 2, 0x44)?;
            Err("selection failed".into())
        })
        .is_err());
    assert_eq!(sequence.position(), 0);
    assert!(!sequence.pending());
    assert_eq!(store.occupied_rows(), 0);
    assert!(sequence
        .prepare_completed(0, 2, |s| execute(s, 1, 0x55))
        .is_err());
    assert!(!sequence.pending());
    assert_eq!(store.occupied_rows(), 0);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = sequence.prepare_completed(0, 2, |s| {
            execute(s, 2, 0x66).unwrap();
            panic!("executor panic")
        });
    }));
    assert!(panic.is_err());
    assert!(!sequence.pending());
    assert_eq!(store.occupied_rows(), 0);
    sequence
        .prepare_completed(0, 2, |s| execute(s, 2, 0x77))
        .unwrap()
        .commit()
        .unwrap();
    assert_eq!(sequence.position(), 2);
}

#[test]
fn generation_acceptance_cancellation_and_invalid_selection_control_numerical_publication() {
    use seismic_engine::{
        generation::{Generation, Options, Readiness, Sampling},
        inputs::InputLayout,
    };
    use std::collections::BTreeSet;
    for outcome in ["accept", "cancel", "bad selection"] {
        let store = store();
        let sequence = OwnedSequence::new(store.create().unwrap());
        let mut generation = Generation::new(
            vec![TokenId(1), TokenId(2)],
            InputLayout::new(2, vec![]).unwrap(),
            Options {
                max_tokens: 8,
                output_capacity: 4,
                context_limit: 16,
                vocabulary: 16,
                stop_tokens: BTreeSet::from([TokenId(15)]),
                sampling: Sampling::Greedy,
                seed: 0,
                forced_quantum: 0,
            },
            None,
        )
        .unwrap();
        let Readiness::Ready(proposal) = generation.ready(2).unwrap() else {
            panic!("expected prefill")
        };
        let row = sequence
            .prepare_completed(proposal.position(), proposal.tokens().len(), |s| {
                execute(s, 2, 0x42)?;
                Ok(Some(TokenId(if outcome == "bad selection" {
                    16
                } else {
                    7
                })))
            })
            .unwrap();
        generation.attach(proposal, row).unwrap();
        if outcome == "cancel" {
            generation.cancel();
        }
        assert_eq!(generation.reconcile().is_err(), outcome == "bad selection");
        if outcome == "accept" {
            assert_eq!(sequence.position(), 2);
            assert_eq!(generation.processed(), 2);
            assert_eq!(generation.generated(), &[TokenId(7)]);
            assert_eq!(values(&sequence), [0x42; 16]);
        } else {
            assert_eq!(sequence.position(), 0);
            assert_eq!(generation.processed(), 0);
            assert!(generation.generated().is_empty());
            assert_eq!(values(&sequence), [0; 16]);
            assert_eq!(store.occupied_rows(), 0);
        }
        assert!(!sequence.pending());
    }
}

#[test]
fn reclaim_prices_follow_owner_aliases_checkpoint_sets_and_pending_rows() {
    let store = store();
    let sequence = OwnedSequence::new(store.create().unwrap());
    assert_eq!(
        OwnedSequence::reclaimable(&store, &[&sequence]).unwrap(),
        16
    );
    let alias = sequence.clone();
    assert_eq!(OwnedSequence::reclaimable(&store, &[&sequence]).unwrap(), 0);
    drop(alias);
    let checkpoint = sequence.checkpoint().unwrap();
    let fork = OwnedSequence::new(checkpoint.fork());
    assert_eq!(
        OwnedSequence::reclaimable(&store, &[&sequence, &fork]).unwrap(),
        0
    );
    drop(checkpoint);
    assert_eq!(OwnedSequence::reclaimable(&store, &[&sequence]).unwrap(), 0);
    assert_eq!(
        OwnedSequence::reclaimable(&store, &[&sequence, &fork]).unwrap(),
        16
    );
    assert!(OwnedSequence::reclaimable(&store, &[&sequence, &sequence]).is_err());
    let row = sequence
        .prepare_completed(0, 1, |s| execute(s, 1, 1))
        .unwrap();
    assert!(OwnedSequence::reclaimable(&store, &[&sequence]).is_err());
    drop(row);
    drop(fork);
    assert_eq!(
        OwnedSequence::reclaimable(&store, &[&sequence]).unwrap(),
        16
    );
}

#[test]
fn typed_capacity_failure_preserves_accepted_state_and_releases_tentative_claims() {
    use seismic_runtime::Error;
    let device = Rc::new(Device::cpu());
    let store = StateStore::new(
        device.clone(),
        16,
        32,
        vec![16],
        vec![ComponentSpec {
            shape: vec![4],
            dtype: DType::F32,
        }],
    )
    .unwrap();
    let sequence = OwnedSequence::new(store.create().unwrap());
    let before = sequence.checkpoint().unwrap();
    store.history().unwrap();
    let retained = device.memory_usage().charged;
    device.set_memory_limit(Some(retained + 8)).unwrap();
    let failure = sequence.prepare_completed(0, 3, |s| execute(s, 3, 1));
    assert!(matches!(
        failure,
        Err(Error::Capacity {
            required: 16,
            available: 8
        })
    ));
    assert!(!sequence.pending());
    assert_eq!(sequence.position(), 0);
    assert_eq!(store.occupied_rows(), 0);
    assert_eq!(device.memory_usage().charged, retained);
    assert_eq!(values(&sequence), [0; 16]);
    device.set_memory_limit(Some(retained + 16)).unwrap();
    let mut row = sequence
        .prepare_completed(0, 3, |s| execute(s, 3, 1))
        .unwrap();
    assert_eq!(device.memory_usage().charged, retained + 16);
    row.commit().unwrap();
    drop(row);
    // The old accepted values are still charged while the checkpoint owns them.
    assert_eq!(device.memory_usage().charged, retained + 16);
    drop(before);
    assert_eq!(device.memory_usage().charged, retained);
    assert_eq!(sequence.position(), 3);
    drop(sequence);
    assert_eq!(store.release_idle().unwrap(), 512);
    assert_eq!(device.memory_usage().charged, 0);
}

#[test]
fn completed_batch_holds_all_owners_and_accepts_peer_outcomes_independently() {
    use seismic_engine::models::sequence::SequenceWork;
    let store = store();
    let sequences: Vec<_> = (0..3)
        .map(|_| OwnedSequence::new(store.create().unwrap()))
        .collect();
    let work: Vec<_> = sequences
        .iter()
        .enumerate()
        .map(|(i, sequence)| SequenceWork {
            sequence,
            position: 0,
            count: i + 1,
        })
        .collect();
    let mut rows = OwnedSequence::prepare_completed_batch(&work, |states| {
        assert!(sequences.iter().all(OwnedSequence::pending));
        assert!(sequences
            .iter()
            .all(|sequence| sequence.position() == 0 && sequence.checkpoint().is_err()));
        for (i, state) in states.iter_mut().enumerate() {
            execute(state, i + 1, (i + 1) as u8)?;
        }
        Ok(vec![
            Err("empty sampling distribution".into()),
            Ok(Some(TokenId(9))),
            Ok(None),
        ])
    })
    .unwrap();
    assert_eq!(store.occupied_rows(), 6);
    assert!(rows[0].selected().is_err());
    assert!(rows[0].commit().is_err());
    assert_eq!(rows[1].selected().unwrap(), Some(TokenId(9)));
    rows[1].commit().unwrap();
    assert_eq!(sequences[1].position(), 2);
    assert_eq!(values(&sequences[1]), [2; 16]);
    assert_eq!(sequences[0].position(), 0);
    assert_eq!(sequences[2].position(), 0);
    drop(rows); // Failed first member and cancelled third member abort independently.
    assert!(sequences.iter().all(|sequence| !sequence.pending()));
    assert_eq!(store.occupied_rows(), 2);
    assert_eq!(values(&sequences[0]), [0; 16]);
    assert_eq!(values(&sequences[2]), [0; 16]);
}

#[test]
fn batch_preflight_shared_failure_malformed_completion_and_unwind_are_atomic() {
    use seismic_engine::models::sequence::SequenceWork;
    let store = store();
    let first = OwnedSequence::new(store.create().unwrap());
    let second = OwnedSequence::new(store.create().unwrap());
    let alias = first.clone();
    let duplicate = [
        SequenceWork {
            sequence: &first,
            position: 0,
            count: 1,
        },
        SequenceWork {
            sequence: &alias,
            position: 0,
            count: 1,
        },
    ];
    assert!(
        OwnedSequence::prepare_completed_batch(&duplicate, |_| panic!("aliased batch executed"))
            .is_err()
    );
    let stale = [
        SequenceWork {
            sequence: &first,
            position: 0,
            count: 1,
        },
        SequenceWork {
            sequence: &second,
            position: 1,
            count: 1,
        },
    ];
    assert!(
        OwnedSequence::prepare_completed_batch(&stale, |_| panic!("stale batch executed")).is_err()
    );
    assert!(!first.pending() && !second.pending());
    let work = [
        SequenceWork {
            sequence: &first,
            position: 0,
            count: 1,
        },
        SequenceWork {
            sequence: &second,
            position: 0,
            count: 1,
        },
    ];
    assert!(OwnedSequence::prepare_completed_batch(&work, |states| {
        execute(states[0], 1, 1)?;
        Err("native batch failed".into())
    })
    .is_err());
    assert!(OwnedSequence::prepare_completed_batch(&work, |states| {
        execute(states[0], 1, 1)?;
        execute(states[1], 1, 2)?;
        Ok(vec![Ok(None)])
    })
    .is_err());
    assert!(OwnedSequence::prepare_completed_batch(&work, |states| {
        execute(states[0], 1, 1)?;
        Ok(vec![Ok(None), Ok(None)]) // Second member did not advance its extent.
    })
    .is_err());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = OwnedSequence::prepare_completed_batch(&work, |states| {
            execute(states[0], 1, 1)?;
            panic!("failed host preparation")
        });
    }));
    assert!(panic.is_err());
    assert!(!first.pending() && !second.pending());
    assert_eq!(first.position(), 0);
    assert_eq!(second.position(), 0);
    assert_eq!(store.occupied_rows(), 0);
    assert_eq!(values(&first), [0; 16]);
    assert_eq!(values(&second), [0; 16]);
}

#[test]
fn semantic_snapshots_follow_independent_batch_acceptance_and_rollback() {
    use seismic_engine::models::sequence::SequenceWork;
    let store = store();
    let a = OwnedSequence::with_semantics(store.create().unwrap(), vec!["prompt"]);
    let b = OwnedSequence::with_semantics(store.create().unwrap(), vec!["prompt"]);
    let checkpoint = b.checkpoint_with_semantics().unwrap();
    let work = [
        SequenceWork {
            sequence: &a,
            position: 0,
            count: 2,
        },
        SequenceWork {
            sequence: &b,
            position: 0,
            count: 2,
        },
    ];
    let mut rows = OwnedSequence::prepare_completed_batch_with_semantics(&work, |states| {
        assert!(a.checkpoint_with_semantics().is_err());
        assert!(b.checkpoint_with_semantics().is_err());
        for (state, semantics) in states {
            execute(state, 2, 0x31)?;
            semantics.push("consumed");
        }
        Ok(vec![Err("empty distribution".into()), Ok(Some(TokenId(7)))])
    })
    .unwrap();
    assert_eq!((a.position(), b.position()), (0, 0));
    assert!(rows[0].commit().is_err());
    rows[1].commit().unwrap();
    drop(rows);
    assert_eq!(a.checkpoint_with_semantics().unwrap().1, vec!["prompt"]);
    let (numeric, semantic) = b.checkpoint_with_semantics().unwrap();
    assert_eq!(numeric.position(), 2);
    assert_eq!(semantic, vec!["prompt", "consumed"]);
    let restored = OwnedSequence::with_semantics(checkpoint.0.fork(), checkpoint.1);
    assert_eq!(restored.position(), 0);
    assert_eq!(
        restored.checkpoint_with_semantics().unwrap().1,
        vec!["prompt"]
    );
    for unwind in [false, true] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            OwnedSequence::prepare_completed_batch_with_semantics(
                &[SequenceWork {
                    sequence: &a,
                    position: 0,
                    count: 1,
                }],
                |states| {
                    let (state, semantics) = &mut states[0];
                    execute(state, 1, 0x72)?;
                    semantics.push("must roll back");
                    if unwind {
                        panic!("injected semantic transaction unwind");
                    }
                    Err("injected semantic transaction error".into())
                },
            )
        }));
        assert!(match result {
            Ok(result) => result.is_err(),
            Err(_) => unwind,
        });
        assert!(!a.pending());
        let (state, semantics) = a.checkpoint_with_semantics().unwrap();
        assert_eq!(state.position(), 0);
        assert_eq!(semantics, vec!["prompt"]);
    }
}
