use magnitude_model_state::{
    BankCapacity, CodecSpec, ComponentDescriptor, ComponentSpec, Holder, LayerRef,
    OwnedAdvanceResolution, OwnedStateAdvance, SequenceState, StateStore,
};
use seismic::{BackendName, DType, Device, DeviceCatalog, Tensor};
use std::rc::Rc;
fn device() -> Device {
    DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap()
}
fn write(tensor: &Tensor, bytes: &[u8]) -> Result<(), magnitude_model_state::Error> {
    let mut tensor = tensor.clone();
    tensor.write_from_host(bytes)?;
    Ok(())
}
fn read(tensor: &Tensor) -> Vec<u8> {
    tensor.read_to_host().unwrap()
}
/// The one-bank row of the store's single recurrent arena.
fn bank(store: &StateStore, index: usize) -> Tensor {
    store.recurrent_arenas()[0]
        .slice_leading(index as u64, index as u64 + 1)
        .unwrap()
}
fn history_component(width: usize, dtype: DType) -> ComponentDescriptor {
    ComponentDescriptor::new(
        LayerRef::Target(0),
        CodecSpec::dense(dtype, width, width),
        1,
    )
    .unwrap()
}
fn store(history: bool, values: bool) -> Rc<StateStore> {
    StateStore::new(
        Rc::new(device()),
        16,
        32,
        if history {
            vec![history_component(4, DType::F32)]
        } else {
            vec![]
        },
        if values {
            vec![ComponentSpec {
                shape: vec![4],
                dtype: DType::F32,
            }]
        } else {
            vec![]
        },
        BankCapacity {
            active: 32,
            in_flight: 32,
            retained: 0,
        },
    )
    .unwrap()
}
fn accept(store: &StateStore, state: SequenceState, count: usize) -> SequenceState {
    let advance = OwnedStateAdvance::begin(state, count).ok().unwrap();
    if store.has_recurrent_components() {
        let successor = bank(store, advance.bindings().following_bank);
        write(&successor, &vec![0; successor.byte_len() as usize]).unwrap();
    }
    let OwnedAdvanceResolution::Committed(next) = advance.commit_all().ok().unwrap() else {
        panic!("full advance must commit");
    };
    next
}
#[test]
#[ignore = "requires a Metal device"]
fn shared_prefix_private_tail_and_parent_first_drop() {
    let store = store(true, true);
    let mut parent = store.create().unwrap();
    parent = accept(&store, parent, 8);
    let checkpoint = parent.checkpoint();
    let mut branches = (0..6).map(|_| checkpoint.fork()).collect::<Vec<_>>();
    assert_eq!(store.occupied_rows(), 8);
    assert!(branches.iter().all(|b| b.history_ranges() == [(0, 8)]));
    parent = accept(&store, parent, 2);
    let branch = branches.remove(0);
    branches.insert(0, accept(&store, branch, 3));
    assert_eq!(checkpoint.position(), 8);
    // The parent grew in place into [8, 10); the branch starts mid-hole.
    assert_eq!(branches[0].history_ranges(), [(0, 8), (19, 3)]);
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
#[ignore = "requires a Metal device"]
fn failed_and_aborted_work_cannot_publish_or_recycle_early() {
    let store = store(true, true);
    let state = store.create().unwrap();
    let original = bank(&store, state.bank_index());
    let before = read(&original);
    let advance = OwnedStateAdvance::begin(state, 5).ok().unwrap();
    let bindings = advance.bindings();
    assert_eq!(bindings.destinations, [0, 1, 2, 3, 4]);
    assert_eq!(store.occupied_rows(), 5);
    write(&bank(&store, bindings.following_bank), &[0xff; 16]).unwrap();
    write(
        &bindings.history[0].buffer.slice_leading(0, 1).unwrap(),
        &[0x33; 16],
    )
    .unwrap();
    // A failed physical submission returns ownership for abort, never commit.
    assert_eq!(store.occupied_rows(), 5);
    let state = advance.abort();
    assert_eq!(state.position(), 0);
    assert_eq!(store.occupied_rows(), 0);
    let after = read(&original);
    assert_eq!(before, after);
    let advance = OwnedStateAdvance::begin(state, 5).ok().unwrap();
    write(
        &bank(&store, advance.bindings().following_bank),
        &[0x22; 16],
    )
    .unwrap();
    let state = advance.abort();
    assert_eq!(read(&bank(&store, state.bank_index())), before);
    assert_eq!(state.position(), 0);
    assert_eq!(store.occupied_rows(), 0);
    let advance = OwnedStateAdvance::begin(state, 5).ok().unwrap();
    assert_eq!(advance.bindings().destinations, [0, 1, 2, 3, 4]);
}
#[test]
#[ignore = "requires a Metal device"]
fn accepted_component_versions_survive_checkpoint_and_fork() {
    let store = store(false, true);
    let parent = store.create().unwrap();
    let checkpoint = parent.checkpoint();
    let child = checkpoint.fork();
    let advance = OwnedStateAdvance::begin(parent, 1).ok().unwrap();
    assert_ne!(advance.bindings().following_bank, child.bank_index());
    write(
        &bank(&store, advance.bindings().following_bank),
        &[0x22; 16],
    )
    .unwrap();
    let OwnedAdvanceResolution::Committed(parent) = advance.commit_all().ok().unwrap() else {
        panic!("full advance must commit");
    };
    let parent_bytes = read(&bank(&store, parent.bank_index()));
    let child_bytes = read(&bank(&store, child.bank_index()));
    assert_eq!(parent_bytes, [0x22; 16]);
    assert_eq!(child_bytes, [0; 16]);
    assert_eq!(store.occupied_rows(), 0);
}
#[test]
#[ignore = "requires a Metal device"]
fn fragmented_reservation_and_capacity_failure_are_atomic() {
    let store = store(true, false);
    let a = accept(&store, store.create().unwrap(), 8);
    let b = accept(&store, store.create().unwrap(), 8);
    let c = accept(&store, store.create().unwrap(), 8);
    assert_eq!(b.history_ranges(), [(16, 8)]);
    // Placement: a [0, 8), b mid-hole [16, 24), c [8, 16).
    drop(c);
    let advance = OwnedStateAdvance::begin(a, 8).ok().unwrap();
    assert_eq!(advance.bindings().destinations, (8..16).collect::<Vec<_>>());
    let _a = advance.abort();
    assert_eq!(store.occupied_rows(), 16);
    let d = store.create().unwrap();
    let advance = OwnedStateAdvance::begin(d, 16).ok().unwrap();
    assert_eq!(
        advance.bindings().destinations,
        (8..16).chain(24..32).collect::<Vec<_>>()
    );
    let OwnedAdvanceResolution::Committed(d) = advance.commit_all().ok().unwrap() else {
        panic!("full advance must commit");
    };
    assert_eq!(store.occupied_rows(), 32);
    let e = store.create().unwrap();
    let Err((e, _)) = OwnedStateAdvance::begin(e, 1) else {
        panic!("exhausted arena must reject reservation");
    };
    assert_eq!(e.position(), 0);
    assert_eq!(store.occupied_rows(), 32);
    drop(d);
    assert_eq!(store.occupied_rows(), 16);
}
#[test]
#[ignore = "requires a Metal device"]
fn trim_preserves_checkpoint_logical_history_and_position() {
    let store = store(true, true);
    let mut parent = store.create().unwrap();
    parent = accept(&store, parent, 8);
    let checkpoint = parent.checkpoint();
    parent.trim_history(5).unwrap();
    assert_eq!(parent.position(), 8);
    assert_eq!(parent.history_ranges(), [(5, 3)]);
    let descendant = parent.checkpoint();
    let fork = descendant.fork();
    let original = checkpoint.fork();
    assert_eq!(fork.history_ranges(), [(5, 3)]);
    assert_eq!(original.history_ranges(), [(0, 8)]);
    parent = accept(&store, parent, 2);
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
#[ignore = "requires a Metal device"]
fn adjacent_claims_merge_but_checkpoint_boundaries_do_not_grow() {
    let store = store(true, false);
    let mut a = store.create().unwrap();
    a = accept(&store, a, 4);
    let cp = a.checkpoint();
    let mut b = cp.fork();
    drop(a);
    b = accept(&store, b, 2);
    // The fork grew in place and joined its rows; the checkpoint still sees 4.
    assert_eq!(b.history_ranges(), [(0, 6)]);
    assert_eq!(cp.fork().history_ranges(), [(0, 4)]);
    drop(cp);
    let cp = b.checkpoint();
    let mut c = cp.fork();
    drop(b);
    drop(cp);
    assert_eq!(c.position(), 6);
    assert_eq!(c.history_ranges(), [(0, 6)]);
    // Trimming releases exactly the trimmed rows.
    c.trim_history(4).unwrap();
    assert_eq!(store.occupied_rows(), 2);
    c = accept(&store, c, 1);
    assert_eq!(c.history_ranges(), [(4, 3)]);
    assert_eq!(store.occupied_rows(), 3);
}
#[test]
#[ignore = "requires a Metal device"]
fn idle_arena_release_and_value_only_or_history_only_sequences() {
    for (history, values) in [(true, false), (false, true), (true, true)] {
        let store = store(history, values);
        let mut parent = store.create().unwrap();
        // History backing is committed on demand, not at creation.
        assert!(store.history_planes().unwrap().is_empty());
        assert_eq!(store.release_idle().unwrap(), 0);
        parent = accept(&store, parent, 4);
        let old = store.history_planes().unwrap();
        let checkpoint = parent.checkpoint();
        let mut branch = checkpoint.fork();
        branch = accept(&store, branch, 2);
        assert_eq!(parent.position(), 4);
        assert_eq!(branch.position(), 6);
        assert_eq!(store.occupied_rows(), if history { 6 } else { 0 });
        drop(parent);
        drop(branch);
        drop(checkpoint);
        // Explicitly retained physical pins keep their bytes and stay usable
        // after logical release.
        assert_eq!(store.release_idle().unwrap(), 0);
        for buffer in &old {
            let bytes = read(&buffer.buffer.slice_leading(0, 4).unwrap());
            assert!(!bytes.is_empty());
        }
        assert!(store.history_planes().unwrap().is_empty());
        drop(old);
        // Without pins, an idle store returns its committed history.
        parent = accept(&store, store.create().unwrap(), 4);
        drop(parent);
        assert_eq!(
            store.release_idle().unwrap(),
            if history { 1024 } else { 0 }
        );
    }
}
#[test]
#[ignore = "requires a Metal device"]
fn context_and_anticipation_bounds() {
    let store = store(true, true);
    let mut state = store.create().unwrap();
    state.anticipate(12).unwrap();
    state.anticipate(3).unwrap();
    assert_eq!(state.expected_end(), 12);
    assert!(state.anticipate(17).is_err());
    let Err((state, _)) = OwnedStateAdvance::begin(state, 0) else {
        panic!("zero-row advance must fail");
    };
    let Err((mut state, _)) = OwnedStateAdvance::begin(state, 17) else {
        panic!("advance beyond context must fail");
    };
    state = accept(&store, state, 16);
    let Err((mut state, _)) = OwnedStateAdvance::begin(state, 1) else {
        panic!("full context must reject another row");
    };
    assert!(state.trim_history(17).is_err());
}

#[test]
#[ignore = "requires a Metal device"]
fn reclamation_counts_selected_handles_once_and_respects_checkpoint_pins() {
    let store = store(true, true);
    let reclaimable = |states: &[&SequenceState]| {
        store
            .exclusive_bytes(
                &states
                    .iter()
                    .map(|state| Holder::State(state))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    };
    // One history row (32 bytes) and one bank (16 bytes).
    let seed = store.create().unwrap();
    // The zero seed is shared by every fresh sequence and never reclaimed.
    assert_eq!(reclaimable(&[&seed]), 0);
    let parent = accept(&store, seed, 1);
    let checkpoint = parent.checkpoint();
    let fork = checkpoint.fork();
    assert_eq!(reclaimable(&[&parent, &fork]), 0);
    drop(checkpoint);
    assert_eq!(reclaimable(&[&parent]), 0);
    assert_eq!(reclaimable(&[&parent, &fork, &parent]), 48);
    // An in-flight advance from the fork pins the shared row and bank.
    let fork = OwnedStateAdvance::begin(fork, 1).ok().unwrap();
    assert_eq!(reclaimable(&[&parent]), 0);
    let fork = fork.abort();
    drop(parent);
    assert_eq!(reclaimable(&[&fork]), 48);
    let other = StateStore::new(
        Rc::new(device()),
        16,
        32,
        vec![],
        vec![],
        BankCapacity {
            active: 1,
            in_flight: 1,
            retained: 0,
        },
    )
    .unwrap();
    assert!(other.exclusive_bytes(&[Holder::State(&fork)]).is_err());
}

#[test]
#[ignore = "requires a Metal device"]
fn owned_advances_reconcile_independently_after_shared_completion() {
    let store = store(true, true);
    let first = store.create().unwrap();
    let second = store.create().unwrap();
    let first_advance = OwnedStateAdvance::begin(first, 2).ok().unwrap();
    let second_advance = OwnedStateAdvance::begin(second, 1).ok().unwrap();
    let first_bindings = first_advance.bindings();
    let second_bindings = second_advance.bindings();
    assert!(first_bindings.history[0]
        .buffer
        .shares_allocation(&second_bindings.history[0].buffer));
    assert_ne!(
        first_bindings.following_bank,
        second_bindings.following_bank
    );
    assert!(first_bindings.recurrent[0].shares_allocation(&second_bindings.recurrent[0]));
    assert!(first_bindings
        .destinations
        .iter()
        .all(|row| !second_bindings.destinations.contains(row)));
    write(&bank(&store, first_bindings.following_bank), &[1; 16]).unwrap();
    write(&bank(&store, second_bindings.following_bank), &[2; 16]).unwrap();
    let OwnedAdvanceResolution::Committed(first) = first_advance.commit_all().ok().unwrap() else {
        panic!("completed first row must commit");
    };
    let second = second_advance.abort();
    assert_eq!(first.position(), 2);
    assert_eq!(second.position(), 0);
    assert_eq!(store.occupied_rows(), 2);
    assert_eq!(read(&bank(&store, first.bank_index())), [1; 16]);
    assert_eq!(read(&bank(&store, second.bank_index())), [0; 16]);

    // A launch provisions backing for all of its advances before they begin.
    store
        .provision(&[first.demand(1), second.demand(1)], 2)
        .unwrap();
    let first_advance = OwnedStateAdvance::begin(first, 1).ok().unwrap();
    let second_advance = OwnedStateAdvance::begin(second, 1).ok().unwrap();
    write(
        &bank(&store, first_advance.bindings().following_bank),
        &[3; 16],
    )
    .unwrap();
    // A shared failed submission aborts both owned advances.
    let first = first_advance.abort();
    let second = second_advance.abort();
    assert_eq!(first.position(), 2);
    assert_eq!(second.position(), 0);
    assert_eq!(store.occupied_rows(), 2);
}
