use magnitude_state::{
    BankCapacity, CodecSpec, ComponentDescriptor, ComponentSpec, HistoryDomainId,
    HistoryDomainLayout, HistoryDomainPlan, Holder, LayerRef, OwnedAdvanceResolution,
    OwnedStateAdvance, OwnedTailRelocation, SLAB_ROW_TILE, SequenceState, StateStore,
    StoreBindings,
};

/// The one history domain of these stores.
const TOKEN: HistoryDomainId = HistoryDomainId(0);
/// The rows of a history page of these stores: the smallest page, as their
/// short histories keep every span limit with it.
const PAGE: usize = SLAB_ROW_TILE;

/// Rows referenced in every history domain of the store.
fn occupied(store: &StateStore) -> usize {
    store
        .history_domains()
        .map(|domain| store.occupied_rows(domain))
        .sum()
}
use seismic::{BackendName, DType, Device, Tensor};
use std::rc::Rc;
fn device() -> Device {
    seismic::Devices::discover(&[])
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap()
}
fn write(tensor: &Tensor, bytes: &[u8]) -> Result<(), magnitude_state::Error> {
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
/// A store of histories of at most 16 rows, with four history pages: as
/// many histories as pages can each take one.
fn store(history: bool, values: bool) -> StoreBindings {
    store_with(history, values, 16, 4)
}
/// A store of histories of at most `context` rows, with `pages` history
/// pages.
fn store_with(history: bool, values: bool, context: usize, pages: usize) -> StoreBindings {
    StateStore::new(
        Rc::new(device()),
        context,
        16,
        if history {
            vec![HistoryDomainPlan {
                layout: HistoryDomainLayout::Token {
                    components: vec![history_component(4, DType::F32)],
                },
                logical_rows: pages * PAGE,
            }]
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
    .1
}
/// An advance of `count` rows, or the panic of its refusal.
fn begin(state: SequenceState, count: usize) -> OwnedStateAdvance {
    OwnedStateAdvance::begin(state, count)
        .unwrap_or_else(|(_, error)| panic!("an advance of {count} rows was refused: {error}"))
}
/// `state`'s blocked last page of `domain` moved to a free page of its own:
/// its rows copied on the host, then the relocation committed.
fn relocate(state: SequenceState, domain: HistoryDomainId) -> SequenceState {
    let relocation = OwnedTailRelocation::prepare(state, domain)
        .unwrap_or_else(|(_, error)| panic!("the relocation was refused: {error}"));
    let copy = relocation.copy();
    for plane in copy.copies() {
        let tensor = &copy.planes()[plane.plane_index];
        for (&from, &to) in plane.from.iter().zip(&plane.to) {
            let row = read(&tensor.slice_leading(from as u64, from as u64 + 1).unwrap());
            write(&tensor.slice_leading(to as u64, to as u64 + 1).unwrap(), &row).unwrap();
        }
    }
    relocation.commit()
}
fn accept(store: &StateStore, state: SequenceState, count: usize) -> SequenceState {
    let advance = begin(state, count);
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
fn shared_prefix_private_tail_and_parent_first_drop() {
    let store = store(true, true);
    let mut parent = store.create().unwrap();
    parent = accept(&store, parent, 8);
    let checkpoint = parent.checkpoint();
    // The parent grows in place within its last page: a checkpoint, unlike
    // a live fork ending there, takes no row.
    parent = accept(&store, parent, 2);
    assert_eq!(parent.history_ranges(TOKEN), [(0, 10)]);
    let mut branches = (0..6).map(|_| checkpoint.fork()).collect::<Vec<_>>();
    assert_eq!(occupied(&store), 10);
    assert!(branches.iter().all(|b| b.history_ranges(TOKEN) == [(0, 8)]));
    // A branch whose last page the parent continued relocates that page's
    // rows to a free page of its own before it grows: refused until then.
    let branch = branches.remove(0);
    assert!(branch.blocked_tails().contains(&TOKEN));
    let Err((branch, _)) = OwnedStateAdvance::begin(branch, 3) else {
        panic!("a blocked last page must relocate before it grows");
    };
    let branch = accept(&store, relocate(branch, TOKEN), 3);
    assert_eq!(checkpoint.position(), 8);
    // Its copied prefix and its new rows are one run from the start of a
    // page apart from the parent's.
    let &[(start, rows)] = branch.history_ranges(TOKEN).as_slice() else {
        panic!("a relocated history holds one run");
    };
    assert_eq!((start % PAGE, rows), (0, 11));
    assert!(start >= PAGE);
    branches.insert(0, branch);
    assert_eq!(branches[1].history_ranges(TOKEN), [(0, 8)]);
    assert_eq!(occupied(&store), 21);
    drop(parent);
    drop(checkpoint);
    assert_eq!(branches[0].position(), 11);
    drop(branches);
    assert_eq!(occupied(&store), 0);
    assert!(store.idle());
}
#[test]
fn failed_and_aborted_work_cannot_publish_or_recycle_early() {
    let store = store(true, true);
    let state = store.create().unwrap();
    let original = bank(&store, state.bank_index());
    let before = read(&original);
    let advance = OwnedStateAdvance::begin(state, 5).ok().unwrap();
    let bindings = advance.bindings();
    assert_eq!(bindings.destinations[0], [0, 1, 2, 3, 4]);
    assert_eq!(occupied(&store), 5);
    write(&bank(&store, bindings.following_bank), &[0xff; 16]).unwrap();
    write(
        &bindings.history[0].buffer.slice_leading(0, 1).unwrap(),
        &[0x33; 16],
    )
    .unwrap();
    // A failed physical submission returns ownership for abort, never commit.
    assert_eq!(occupied(&store), 5);
    let state = advance.abort();
    assert_eq!(state.position(), 0);
    assert_eq!(occupied(&store), 0);
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
    assert_eq!(occupied(&store), 0);
    let advance = OwnedStateAdvance::begin(state, 5).ok().unwrap();
    assert_eq!(advance.bindings().destinations[0], [0, 1, 2, 3, 4]);
}
#[test]
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
    assert_eq!(occupied(&store), 0);
}
#[test]
fn fragmented_reservation_and_capacity_failure_are_atomic() {
    // Three pages, and histories long enough to outgrow one.
    let store = store_with(true, false, 2 * PAGE, 3);
    let a = accept(&store, store.create().unwrap(), PAGE - 6);
    let b = accept(&store, store.create().unwrap(), 8);
    let c = accept(&store, store.create().unwrap(), 8);
    // A fresh history takes the free run at row 0 from its start, else the
    // middle page of the largest free run.
    assert_eq!(a.history_ranges(TOKEN), [(0, PAGE - 6)]);
    assert_eq!(b.history_ranges(TOKEN), [(PAGE, 8)]);
    assert_eq!(c.history_ranges(TOKEN), [(2 * PAGE, 8)]);
    // a's last page holds 6 more rows and no page is free: 16 more are
    // refused, and nothing changes.
    let Err((a, _)) = OwnedStateAdvance::begin(a, 16) else {
        panic!("an advance past every free row must be refused");
    };
    assert_eq!(a.position(), PAGE - 6);
    assert_eq!(occupied(&store), PAGE + 10);
    // With c's page free, a's 16 rows are one reservation over two holes:
    // in place to its page's end, then that page.
    drop(c);
    let destinations = (PAGE - 6..PAGE).chain(2 * PAGE..2 * PAGE + 10).collect::<Vec<_>>();
    let advance = begin(a, 16);
    assert_eq!(advance.bindings().destinations[0], destinations);
    let a = advance.abort();
    assert_eq!(occupied(&store), PAGE + 2);
    let advance = begin(a, 16);
    assert_eq!(advance.bindings().destinations[0], destinations);
    let OwnedAdvanceResolution::Committed(a) = advance.commit_all().ok().unwrap() else {
        panic!("full advance must commit");
    };
    assert_eq!(a.history_ranges(TOKEN), [(0, PAGE), (2 * PAGE, 10)]);
    assert_eq!(occupied(&store), PAGE + 18);
    // No page is free: a fresh history is refused, and nothing changes.
    let e = store.create().unwrap();
    let Err((e, _)) = OwnedStateAdvance::begin(e, 1) else {
        panic!("exhausted arena must reject reservation");
    };
    assert_eq!(e.position(), 0);
    assert_eq!(occupied(&store), PAGE + 18);
    drop(a);
    assert_eq!(occupied(&store), 8);
    drop(b);
}
#[test]
fn trim_preserves_checkpoint_logical_history_and_position() {
    let store = store(true, true);
    let mut parent = store.create().unwrap();
    parent = accept(&store, parent, 8);
    let checkpoint = parent.checkpoint();
    parent.trim_history(5).unwrap();
    assert_eq!(parent.position(), 8);
    assert_eq!(parent.history_ranges(TOKEN), [(5, 3)]);
    let descendant = parent.checkpoint();
    let fork = descendant.fork();
    let original = checkpoint.fork();
    assert_eq!(fork.history_ranges(TOKEN), [(5, 3)]);
    assert_eq!(original.history_ranges(TOKEN), [(0, 8)]);
    // Live forks ending at the parent's row would make it relocate; the
    // checkpoints, which keep its rows, do not.
    drop(fork);
    drop(original);
    parent = accept(&store, parent, 2);
    parent.trim_history(8).unwrap();
    assert_eq!(parent.history_ranges(TOKEN), [(8, 2)]);
    assert_eq!(occupied(&store), 10);
    drop(checkpoint);
    drop(descendant);
    assert_eq!(occupied(&store), 2);
    parent.trim_history(10).unwrap();
    assert_eq!(occupied(&store), 0);
    assert_eq!(parent.position(), 10);
}
#[test]
fn adjacent_claims_merge_but_checkpoint_boundaries_do_not_grow() {
    let store = store(true, false);
    let mut a = store.create().unwrap();
    a = accept(&store, a, 4);
    let cp = a.checkpoint();
    let mut b = cp.fork();
    drop(a);
    b = accept(&store, b, 2);
    // The fork grew in place and joined its rows; the checkpoint still sees 4.
    assert_eq!(b.history_ranges(TOKEN), [(0, 6)]);
    assert_eq!(cp.fork().history_ranges(TOKEN), [(0, 4)]);
    drop(cp);
    let cp = b.checkpoint();
    let mut c = cp.fork();
    drop(b);
    drop(cp);
    assert_eq!(c.position(), 6);
    assert_eq!(c.history_ranges(TOKEN), [(0, 6)]);
    // Trimming releases exactly the trimmed rows.
    c.trim_history(4).unwrap();
    assert_eq!(occupied(&store), 2);
    c = accept(&store, c, 1);
    assert_eq!(c.history_ranges(TOKEN), [(4, 3)]);
    assert_eq!(occupied(&store), 3);
}
#[test]
fn idle_arena_release_and_value_only_or_history_only_sequences() {
    for (history, values) in [(true, false), (false, true), (true, true)] {
        let mut store = store(history, values);
        let mut parent = store.create().unwrap();
        // A store starts with one backed history slab per history domain.
        assert_eq!(store.history_planes().unwrap().is_empty(), !history);
        assert_eq!(store.release_idle().unwrap(), 0);
        parent = accept(&store, parent, 4);
        let old = store.history_planes().unwrap();
        let checkpoint = parent.checkpoint();
        let mut branch = checkpoint.fork();
        // A branch ending at its live parent's row relocates it first.
        if history {
            branch = relocate(branch, TOKEN);
        }
        branch = accept(&store, branch, 2);
        assert_eq!(parent.position(), 4);
        assert_eq!(branch.position(), 6);
        assert_eq!(occupied(&store), if history { 10 } else { 0 });
        drop(parent);
        drop(branch);
        drop(checkpoint);
        // Slab placement cannot change under a live binding of its planes:
        // releasing them is refused until the binding is dropped.
        for buffer in &old {
            let bytes = read(&buffer.buffer.slice_leading(0, 4).unwrap());
            assert!(!bytes.is_empty());
        }
        assert_eq!(store.release_idle().is_err(), history);
        drop(old);
        store.release_idle().unwrap();
        assert!(store.history_planes().unwrap().is_empty());
        // An idle store returns its committed history. Advances never grow
        // the store: the binding right provisions first.
        parent = store.create().unwrap();
        store.provision(&parent.demands(4), 1).unwrap();
        parent = accept(&store, parent, 4);
        drop(parent);
        // Its one history slab, the full slab byte target.
        assert_eq!(
            store.release_idle().unwrap() as u64,
            if history {
                magnitude_state::SLAB_BYTE_TARGET
            } else {
                0
            }
        );
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
    // An in-flight advance from the fork pins the shared row and bank. A
    // checkpoint holds them with it here: a live parent ending at that row
    // would make the fork relocate it rather than share it.
    let pinned = parent.checkpoint();
    drop(parent);
    let fork = OwnedStateAdvance::begin(fork, 1).ok().unwrap();
    assert_eq!(store.exclusive_bytes(&[Holder::Checkpoint(&pinned)]).unwrap(), 0);
    let fork = fork.abort();
    drop(pinned);
    assert_eq!(reclaimable(&[&fork]), 48);
    let other = StateStore::new(
        Rc::new(device()),
        16,
        16,
        vec![],
        vec![],
        BankCapacity {
            active: 1,
            in_flight: 1,
            retained: 0,
        },
    )
    .unwrap()
    .0;
    assert!(other.exclusive_bytes(&[Holder::State(&fork)]).is_err());
}

#[test]
fn owned_advances_reconcile_independently_after_shared_completion() {
    let mut store = store(true, true);
    let first = store.create().unwrap();
    let second = store.create().unwrap();
    let first_advance = begin(first, 2);
    let second_advance = begin(second, 1);
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
    assert!(first_bindings.destinations[0]
        .iter()
        .all(|row| !second_bindings.destinations[0].contains(row)));
    write(&bank(&store, first_bindings.following_bank), &[1; 16]).unwrap();
    write(&bank(&store, second_bindings.following_bank), &[2; 16]).unwrap();
    let OwnedAdvanceResolution::Committed(first) = first_advance.commit_all().ok().unwrap() else {
        panic!("completed first row must commit");
    };
    let second = second_advance.abort();
    assert_eq!(first.position(), 2);
    assert_eq!(second.position(), 0);
    assert_eq!(occupied(&store), 2);
    assert_eq!(read(&bank(&store, first.bank_index())), [1; 16]);
    assert_eq!(read(&bank(&store, second.bank_index())), [0; 16]);

    // A launch provisions backing for all of its advances before they begin.
    store
        .provision(&[first.demands(1), second.demands(1)].concat(), 2)
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
    assert_eq!(occupied(&store), 2);
}
