use seismic_engine::state::{ComponentSpec, SequenceState, StateStore};
use seismic_lang::types::DType;
use seismic_runtime::Device;
use std::rc::Rc;
fn store(history: bool, values: bool) -> Rc<StateStore> {
    StateStore::new(
        Rc::new(Device::cpu()),
        16,
        32,
        if history { vec![16] } else { vec![] },
        if values {
            vec![ComponentSpec {
                shape: vec![4],
                dtype: DType::F32,
            }]
        } else {
            vec![]
        },
    )
    .unwrap()
}
fn accept(state: &mut SequenceState, count: usize) {
    let mut advance = state.begin(count).unwrap();
    advance
        .execute(|b| {
            for v in b.following {
                v.write(&vec![0; v.len()])?;
            }
            Ok(())
        })
        .unwrap();
    advance.commit().unwrap();
}
#[test]
fn shared_prefix_private_tail_and_parent_first_drop() {
    let store = store(true, true);
    let mut parent = store.create().unwrap();
    accept(&mut parent, 8);
    let checkpoint = parent.checkpoint();
    let mut branches = (0..6).map(|_| checkpoint.fork()).collect::<Vec<_>>();
    assert_eq!(store.occupied_rows(), 8);
    assert!(branches.iter().all(|b| b.history_ranges() == [(0, 8)]));
    accept(&mut parent, 2);
    accept(&mut branches[0], 3);
    assert_eq!(checkpoint.position(), 8);
    assert_eq!(branches[0].history_ranges(), [(0, 8), (10, 3)]);
    assert_eq!(branches[1].history_ranges(), [(0, 8)]);
    assert_eq!(store.occupied_rows(), 13);
    drop(parent);
    drop(checkpoint);
    assert_eq!(branches[0].position(), 11);
    drop(branches);
    assert_eq!(store.occupied_rows(), 0);
    assert!(store.idle());
}
#[test]
fn failed_and_aborted_work_cannot_publish_or_recycle_early() {
    let store = store(true, true);
    let mut state = store.create().unwrap();
    let original = state.values()[0].clone();
    let mut before = [0; 16];
    original.read(&mut before).unwrap();
    let mut advance = state.begin(5).unwrap();
    assert_eq!(advance.destinations(), [0, 1, 2, 3, 4]);
    assert!(advance
        .execute(|b| {
            assert_eq!(store.occupied_rows(), 5);
            b.following[0].write(&[0xff; 16])?;
            b.history[0].write(&[0x33; 16])?;
            Err("failed after physical writes".into())
        })
        .is_err());
    assert_eq!(store.occupied_rows(), 5);
    assert!(advance.commit().is_err());
    assert_eq!(state.position(), 0);
    assert_eq!(store.occupied_rows(), 0);
    let mut after = [0; 16];
    original.read(&mut after).unwrap();
    assert_eq!(before, after);
    let mut advance = state.begin(5).unwrap();
    advance
        .execute(|b| {
            b.following[0].write(&[0x22; 16])?;
            Ok(())
        })
        .unwrap();
    assert!(advance.execute(|_| Ok(())).is_err());
    advance.abort();
    assert_eq!(state.position(), 0);
    assert_eq!(store.occupied_rows(), 0);
    assert!(state.begin(1).unwrap().commit().is_err());
    assert_eq!(state.begin(5).unwrap().destinations(), [0, 1, 2, 3, 4]);
}
#[test]
fn accepted_component_versions_survive_checkpoint_and_fork() {
    let store = store(false, true);
    let mut parent = store.create().unwrap();
    let checkpoint = parent.checkpoint();
    let child = checkpoint.fork();
    let mut advance = parent.begin(1).unwrap();
    advance
        .execute(|b| {
            b.following[0].write(&[0x22; 16])?;
            Ok(())
        })
        .unwrap();
    advance.commit().unwrap();
    let mut parent_bytes = [0; 16];
    let mut child_bytes = [0; 16];
    parent.values()[0].read(&mut parent_bytes).unwrap();
    child.values()[0].read(&mut child_bytes).unwrap();
    assert_eq!(parent_bytes, [0x22; 16]);
    assert_eq!(child_bytes, [0; 16]);
    assert_eq!(store.occupied_rows(), 0);
}
#[test]
fn fragmented_reservation_and_capacity_failure_are_atomic() {
    let store = store(true, false);
    let mut a = store.create().unwrap();
    let mut b = store.create().unwrap();
    let mut c = store.create().unwrap();
    accept(&mut a, 8);
    accept(&mut b, 8);
    accept(&mut c, 8);
    drop(b);
    let advance = a.begin(8).unwrap();
    assert_eq!(advance.destinations(), (8..16).collect::<Vec<_>>());
    advance.abort();
    assert_eq!(store.occupied_rows(), 16);
    let mut d = store.create().unwrap();
    let mut advance = d.begin(16).unwrap();
    assert_eq!(
        advance.destinations(),
        (8..16).chain(24..32).collect::<Vec<_>>()
    );
    advance.execute(|_| Ok(())).unwrap();
    advance.commit().unwrap();
    assert_eq!(store.occupied_rows(), 32);
    let mut e = store.create().unwrap();
    assert!(e.begin(1).is_err());
    assert_eq!(e.position(), 0);
    assert_eq!(store.occupied_rows(), 32);
    drop(d);
    assert_eq!(store.occupied_rows(), 16);
}
#[test]
fn trim_preserves_checkpoint_logical_history_and_position() {
    let store = store(true, true);
    let mut parent = store.create().unwrap();
    accept(&mut parent, 8);
    let checkpoint = parent.checkpoint();
    parent.trim_history(5).unwrap();
    assert_eq!(parent.position(), 8);
    assert_eq!(parent.history_ranges(), [(5, 3)]);
    let descendant = parent.checkpoint();
    let fork = descendant.fork();
    let original = checkpoint.fork();
    assert_eq!(fork.history_ranges(), [(5, 3)]);
    assert_eq!(original.history_ranges(), [(0, 8)]);
    accept(&mut parent, 2);
    parent.trim_history(8).unwrap();
    assert_eq!(parent.history_ranges(), [(8, 2)]);
    assert_eq!(store.occupied_rows(), 10);
    drop(original);
    drop(checkpoint);
    drop(fork);
    drop(descendant);
    assert_eq!(store.occupied_rows(), 2);
    parent.trim_history(10).unwrap();
    assert_eq!(store.occupied_rows(), 0);
    assert_eq!(parent.position(), 10);
}
#[test]
fn exclusive_adjacent_extents_merge_but_checkpoint_boundaries_do_not_grow() {
    let store = store(true, false);
    let mut a = store.create().unwrap();
    accept(&mut a, 4);
    let cp = a.checkpoint();
    let mut b = cp.fork();
    drop(a);
    drop(cp);
    accept(&mut b, 2);
    let cp = b.checkpoint();
    let mut c = cp.fork();
    drop(b);
    drop(cp);
    assert_eq!(c.position(), 6);
    assert_eq!(c.history_ranges(), [(0, 6)]);
    c.trim_history(4).unwrap();
    assert_eq!(store.occupied_rows(), 6);
    accept(&mut c, 1);
    assert_eq!(store.occupied_rows(), 7);
}
#[test]
fn idle_arena_release_and_value_only_or_history_only_sequences() {
    for (history, values) in [(true, false), (false, true), (true, true)] {
        let store = store(history, values);
        let mut parent = store.create().unwrap();
        let old = store.history().unwrap();
        assert_eq!(store.release_idle().unwrap(), 0);
        accept(&mut parent, 4);
        let checkpoint = parent.checkpoint();
        let mut branch = checkpoint.fork();
        accept(&mut branch, 2);
        assert_eq!(parent.position(), 4);
        assert_eq!(branch.position(), 6);
        assert_eq!(store.occupied_rows(), if history { 6 } else { 0 });
        drop(parent);
        drop(branch);
        drop(checkpoint);
        assert_eq!(store.release_idle().unwrap(), 0);
        // Explicitly retained physical pins remain usable after logical release.
        for buffer in old {
            let mut bytes = vec![0; buffer.len()];
            buffer.read(&mut bytes).unwrap();
        }
        assert_eq!(store.history().unwrap().len(), usize::from(history));
        assert_eq!(store.release_idle().unwrap(), if history { 512 } else { 0 });
    }
}
#[test]
fn context_and_anticipation_bounds() {
    let store = store(true, true);
    let mut state = store.create().unwrap();
    state.anticipate(12).unwrap();
    state.anticipate(3).unwrap();
    assert_eq!(state.expected_end(), 12);
    assert!(state.anticipate(17).is_err());
    assert!(state.begin(0).is_err());
    assert!(state.begin(17).is_err());
    accept(&mut state, 16);
    assert!(state.begin(1).is_err());
    assert!(state.trim_history(17).is_err());
}

#[test]
fn reclamation_counts_selected_handles_once_and_respects_checkpoint_pins() {
    let store = store(true, true);
    let parent = store.create().unwrap();
    let checkpoint = parent.checkpoint();
    let fork = checkpoint.fork();
    assert_eq!(store.reclaimable(&[&parent, &fork]).unwrap(), 0);
    drop(checkpoint);
    assert_eq!(store.reclaimable(&[&parent]).unwrap(), 0);
    assert_eq!(store.reclaimable(&[&parent, &fork, &parent]).unwrap(), 16);
    let external = parent.values()[0].view(0..4).unwrap();
    assert_eq!(store.reclaimable(&[&parent, &fork]).unwrap(), 0);
    drop(external);
    drop(parent);
    assert_eq!(store.reclaimable(&[&fork]).unwrap(), 16);
    let other = StateStore::new(Rc::new(Device::cpu()), 16, 32, vec![], vec![]).unwrap();
    assert!(other.reclaimable(&[&fork]).is_err());
}

#[test]
fn shared_execution_publishes_completion_for_all_rows_or_none() {
    use seismic_engine::state::StateAdvance;
    let store = store(true, true);
    let mut first = store.create().unwrap();
    let mut second = store.create().unwrap();
    let mut advances = vec![first.begin(2).unwrap(), second.begin(1).unwrap()];
    StateAdvance::execute_batch(&mut advances, |bindings| {
        assert_eq!(bindings.len(), 2);
        assert!(bindings[0].history[0].shares_allocation(&bindings[1].history[0]));
        assert!(!bindings[0].following[0].shares_allocation(&bindings[1].following[0]));
        assert!(bindings[0]
            .destinations
            .iter()
            .all(|row| !bindings[1].destinations.contains(row)));
        bindings[0].following[0].write(&[1; 16])?;
        bindings[1].following[0].write(&[2; 16])?;
        Ok(())
    })
    .unwrap();
    advances.remove(0).commit().unwrap();
    drop(advances); // An independently cancelled peer never publishes its successor.
    assert_eq!(first.position(), 2);
    assert_eq!(second.position(), 0);
    assert_eq!(store.occupied_rows(), 2);
    let mut bytes = [0; 16];
    first.values()[0].read(&mut bytes).unwrap();
    assert_eq!(bytes, [1; 16]);
    second.values()[0].read(&mut bytes).unwrap();
    assert_eq!(bytes, [0; 16]);
    let mut advances = vec![first.begin(1).unwrap(), second.begin(1).unwrap()];
    assert!(StateAdvance::execute_batch(&mut advances, |bindings| {
        bindings[0].following[0].write(&[3; 16])?;
        Err("shared native completion failed".into())
    })
    .is_err());
    for advance in advances {
        assert!(advance.commit().is_err());
    }
    assert_eq!(first.position(), 2);
    assert_eq!(second.position(), 0);
    assert_eq!(store.occupied_rows(), 2);
}
