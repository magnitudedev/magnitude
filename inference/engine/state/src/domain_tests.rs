//! History domains in one store: window trimming and sharing, per-domain
//! placement, compaction and accounting, and the single Token domain's
//! identity with the one-domain store.

use super::*;
use seismic::{BackendName, DeviceCatalog};

const TOKEN: HistoryDomainId = HistoryDomainId(0);
const WINDOW: HistoryDomainId = HistoryDomainId(1);

fn cpu_device() -> Option<Rc<Device>> {
    DeviceCatalog::discover()
        .ok()
        .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        .map(Rc::new)
}

fn component(layer: u32, width: usize) -> ComponentDescriptor {
    ComponentDescriptor::new(
        LayerRef::Target(layer),
        CodecSpec::dense(DType::F32, width, width),
        1,
    )
    .unwrap()
}

fn banks() -> BankCapacity {
    BankCapacity {
        active: 4,
        in_flight: 4,
        retained: 4,
    }
}

/// A Token domain (layer 0) and a Window(`window`) domain (layer 1) of
/// `width`-wide dense F32 rows.
fn windowed_store(
    device: Rc<Device>,
    context: usize,
    window: usize,
    max_advance: usize,
    window_rows: usize,
    width: usize,
) -> Rc<StateStore> {
    StateStore::new(
        device,
        context,
        max_advance,
        vec![
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Token {
                    components: vec![component(0, 1)],
                },
                logical_rows: context,
            },
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Window {
                    rows: window,
                    components: vec![component(1, width)],
                },
                logical_rows: window_rows,
            },
        ],
        vec![],
        banks(),
    )
    .unwrap()
}

fn commit(state: SequenceState, rows: usize) -> SequenceState {
    let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
    let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
        panic!("an attention advance commits");
    };
    state
}

fn rows_of(ranges: &[(usize, usize)]) -> Vec<usize> {
    ranges
        .iter()
        .flat_map(|&(start, count)| start..start + count)
        .collect()
}

/// Reference exactly `[start, start + count)` of a domain's free rows.
fn take(arena: &Rc<RefCell<Arena>>, start: usize, count: usize) -> Claims {
    let mut inner = arena.borrow_mut();
    let hole = inner
        .free
        .iter()
        .position(|(hole, size)| *hole <= start && start + count <= hole + size)
        .expect("rows lie in one hole");
    let offset = start - inner.free[hole].0;
    inner.take(hole, offset, count);
    drop(inner);
    Claims::new(arena, vec![(start, count)])
}

fn span_within_slab(ranges: &[(usize, usize)], slab_rows: usize) -> bool {
    ranges
        .iter()
        .all(|&(start, count)| start / slab_rows == (start + count - 1) / slab_rows)
}

#[test]
fn window_histories_keep_their_last_rows() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device, 64, 4, 8, 64, 1);
    let state = commit(store.create().unwrap(), 3);
    assert_eq!(state.history_start(WINDOW), 0);
    assert_eq!(store.occupied_rows(WINDOW), 3);
    let advance = OwnedStateAdvance::begin(state, 3).ok().unwrap();
    // Tentative rows are held with the whole retained window.
    assert_eq!(advance.bindings().destinations[1].len(), 3);
    assert_eq!(store.occupied_rows(WINDOW), 6);
    let fresh = advance.bindings().destinations[1].clone();
    let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
        panic!("an attention advance commits");
    };
    // Position 6 keeps rows [2, 6); the Token domain keeps all 6.
    assert_eq!(state.position(), 6);
    assert_eq!(state.history_start(WINDOW), 2);
    assert_eq!(state.history_start(TOKEN), 0);
    assert_eq!(rows_of(&state.history_ranges(TOKEN)).len(), 6);
    let window = rows_of(&state.history_ranges(WINDOW));
    assert_eq!(window.len(), 4);
    assert_eq!(window[1..], fresh[..]);
    assert_eq!(store.occupied_rows(WINDOW), 4);
    assert_eq!(store.occupied_rows(TOKEN), 6);
    // A query at position 6 over a window of 3 tokens reads rows 4 and 5.
    assert_eq!(rows_of(&state.visible_ranges(WINDOW, 4)), fresh[1..]);
    assert_eq!(state.visible_ranges(TOKEN, 0), state.history_ranges(TOKEN));
    // Freed rows are ordinary free space.
    assert_eq!(store.free_rows(WINDOW), 64 - 4);
    assert_eq!(store.available_rows(), 64 - 6);
}

#[test]
fn windows_smaller_than_an_advance_write_every_row_then_trim() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device, 64, 2, 8, 16, 1);
    let advance = OwnedStateAdvance::begin(store.create().unwrap(), 5)
        .ok()
        .unwrap();
    // Every fresh row is written: later rows of the advance attend to it.
    let fresh = advance.bindings().destinations[1].clone();
    assert_eq!(fresh.len(), 5);
    let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
        panic!("an attention advance commits");
    };
    assert_eq!(state.history_start(WINDOW), 3);
    assert_eq!(rows_of(&state.history_ranges(WINDOW)), fresh[3..]);
    assert_eq!(store.occupied_rows(WINDOW), 2);
    // A speculative advance longer than the window keeps only the rows
    // around its accepted prefix; rejected rows return.
    let advance = OwnedStateAdvance::begin_speculative(state, 6, 1)
        .ok()
        .unwrap();
    assert_eq!(store.occupied_rows(WINDOW), 8);
    let fresh = advance.bindings().destinations[1].clone();
    let OwnedAdvanceResolution::Committed(state) = advance.commit(4).ok().unwrap() else {
        panic!("an accepted prefix commits");
    };
    assert_eq!(state.position(), 9);
    assert_eq!(rows_of(&state.history_ranges(WINDOW)), fresh[2..4]);
    assert_eq!(store.occupied_rows(WINDOW), 2);
    assert_eq!(store.occupied_rows(TOKEN), 9);
}

#[test]
fn tentative_window_rows_roll_back() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device, 64, 3, 8, 16, 1);
    let state = commit(store.create().unwrap(), 5);
    let before = (state.domain_ranges(), store.occupied_rows(WINDOW));
    let advance = OwnedStateAdvance::begin(state, 4).ok().unwrap();
    assert_eq!(store.occupied_rows(WINDOW), 3 + 4);
    let state = advance.abort();
    assert_eq!((state.domain_ranges(), store.occupied_rows(WINDOW)), before);
    let advance = OwnedStateAdvance::begin(state, 4).ok().unwrap();
    let OwnedAdvanceResolution::Aborted(state) = advance.commit(0).ok().unwrap() else {
        panic!("an empty prefix aborts");
    };
    assert_eq!((state.domain_ranges(), store.occupied_rows(WINDOW)), before);
}

#[test]
fn checkpoints_share_the_window_rows_without_copying() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device, 64, 4, 8, 32, 1);
    let state = commit(store.create().unwrap(), 6);
    // A checkpoint at 6 references rows [2, 6).
    let checkpoint = state.checkpoint();
    assert_eq!(checkpoint.history_start(WINDOW), 2);
    assert_eq!(
        checkpoint.history_ranges(WINDOW),
        state.history_ranges(WINDOW)
    );
    assert_eq!(store.occupied_rows(WINDOW), 4);
    drop(state);
    assert_eq!(store.occupied_rows(WINDOW), 4);
    // A fork (a resumed request) shares them; its advance trims only its
    // own reference, so the checkpoint keeps every row it saw.
    let fork = commit(checkpoint.fork(), 1);
    assert_eq!(fork.history_start(WINDOW), 3);
    assert_eq!(
        fork.history_ranges(WINDOW)[0].0,
        checkpoint.history_ranges(WINDOW)[0].0 + 1
    );
    assert_eq!(store.occupied_rows(WINDOW), 5);
    let row = store.history_row_bytes(WINDOW) + store.history_row_bytes(TOKEN);
    // The fork alone owns its new row in each domain; the checkpoint alone
    // owns the window row the fork trimmed.
    assert_eq!(store.exclusive_bytes(&[Holder::State(&fork)]).unwrap(), row);
    assert_eq!(
        store
            .exclusive_bytes(&[Holder::Checkpoint(&checkpoint)])
            .unwrap(),
        store.history_row_bytes(WINDOW)
    );
    let census = store
        .holding_census(
            &[Holder::State(&fork)],
            &[Holder::Checkpoint(&checkpoint)],
            &[],
        )
        .unwrap();
    assert_eq!(census.total(), store.committed_bytes());
    assert_eq!(census.retained, store.history_row_bytes(WINDOW));
    drop(checkpoint);
    assert_eq!(store.occupied_rows(WINDOW), 4);
    assert_eq!(store.occupied_rows(TOKEN), 7);
}

#[test]
fn successors_follow_the_trimmed_window() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device, 64, 2, 8, 32, 1);
    let first = OwnedStateAdvance::begin(store.create().unwrap(), 3)
        .ok()
        .unwrap();
    let second = first.successor(1).unwrap();
    // The successor reads what its predecessor keeps: the last 2 rows.
    assert_eq!(second.history_start(WINDOW), 1);
    assert_eq!(
        rows_of(&second.history_ranges(WINDOW)),
        first.bindings().destinations[1][1..]
    );
    let third = second.successor(1).unwrap();
    assert_eq!(third.history_start(WINDOW), 2);
    let OwnedAdvanceResolution::Committed(state) = first.commit_all().ok().unwrap() else {
        panic!("an attention advance commits");
    };
    let second = second.attach(state).ok().unwrap();
    let OwnedAdvanceResolution::Committed(state) = second.commit_all().ok().unwrap() else {
        panic!("an attention advance commits");
    };
    assert_eq!(state.history_start(WINDOW), 2);
    let third = third.attach(state).ok().unwrap();
    let OwnedAdvanceResolution::Committed(state) = third.commit_all().ok().unwrap() else {
        panic!("an attention advance commits");
    };
    assert_eq!((state.position(), state.history_start(WINDOW)), (5, 3));
    assert_eq!(store.occupied_rows(WINDOW), 2);
}

/// One wide window: 256 rows per slab and two slabs of logical rows. A
/// long decode wraps through both slabs; every retained row keeps its
/// position's tag, spans never cross a slab edge, and an emptied slab is
/// freed by the release rule and its index reused.
#[test]
fn window_rows_cross_slab_edges_and_reuse_freed_slabs() {
    let Some(device) = cpu_device() else { return };
    const WINDOW_ROWS: usize = 8;
    // 2 planes of 32768 F32 values: 256 KiB rows, 256 rows per slab.
    let width = 32768;
    let store = windowed_store(device.clone(), 4096, WINDOW_ROWS, 4, 512, width);
    let slab_rows = store.history_slab_rows(WINDOW);
    assert_eq!(slab_rows, SLAB_ROW_TILE);
    assert_eq!(store.span_bound(WINDOW), 1 + 16);
    // Back both slabs, so the window runs from slab 0 straight into slab 1.
    store
        .provision(
            &[RowDemand {
                domain: WINDOW,
                rows: slab_rows + 1,
            }],
            0,
        )
        .unwrap();
    let plane = |store: &StateStore| {
        store
            .history_planes()
            .unwrap()
            .into_iter()
            .find(|plane| plane.domain == WINDOW)
            .unwrap()
            .buffer
    };
    let tag = |position: usize| {
        let mut row = vec![0u8; width * 4];
        row[..8].copy_from_slice(&(position as u64).to_le_bytes());
        row
    };
    let read = |store: &StateStore, row: usize| {
        let bytes = plane(store)
            .slice_leading(row as u64, row as u64 + 1)
            .unwrap()
            .read_to_host()
            .unwrap();
        u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize
    };
    let step = |state: SequenceState, rows: usize| {
        store.provision(&state.demands(rows), 0).unwrap();
        let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
        let buffer = advance.bindings().history
            [advance.bindings().history.iter().position(|p| p.domain == WINDOW).unwrap()]
        .buffer
        .clone();
        for (offset, &row) in advance.bindings().destinations[1].iter().enumerate() {
            buffer
                .slice_leading(row as u64, row as u64 + 1)
                .unwrap()
                .write_from_host(&tag(advance.position() + offset))
                .unwrap();
        }
        drop(buffer);
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("an attention advance commits");
        };
        state
    };
    let check = |state: &SequenceState| {
        let ranges = state.history_ranges(WINDOW);
        assert!(span_within_slab(&ranges, slab_rows));
        assert!(ranges.len() <= store.span_bound(WINDOW));
        let start = state.history_start(WINDOW);
        assert_eq!(start, state.position().saturating_sub(WINDOW_ROWS));
        for (offset, row) in rows_of(&ranges).into_iter().enumerate() {
            assert_eq!(read(&store, row), start + offset, "row {row}");
        }
    };
    let mut state = store.create().unwrap();
    let mut crossed = false;
    for _ in 0..700 {
        state = step(state, 1);
        check(&state);
        crossed |= state.history_ranges(WINDOW).len() > 1;
        // Idle release keeps one spare slab and never copies.
        store
            .shrink_with(ShrinkPolicy::Idle, |_, _| -> Result<(), Error> {
                panic!("idle release never copies")
            })
            .unwrap();
    }
    assert!(crossed, "the window spanned a slab edge");
    assert_eq!(store.occupied_rows(WINDOW), WINDOW_ROWS);

    // Reclaim frees whichever slab the window does not occupy; the next
    // growth adds the lowest unbacked index again.
    let before = device.memory_usage().charged;
    let released = store.shrink(ShrinkPolicy::Reclaim).unwrap();
    let occupied_slabs = state
        .history_ranges(WINDOW)
        .iter()
        .map(|&(start, _)| start / slab_rows)
        .collect::<BTreeSet<_>>();
    let backed = store.domains[WINDOW.0].arena.borrow().backed.clone();
    assert!(backed.len() <= 2);
    assert!(occupied_slabs.is_subset(&backed));
    if released > 0 {
        assert!(device.memory_usage().charged < before);
    }
    check(&state);
    for _ in 0..600 {
        state = step(state, 3);
        check(&state);
    }
    assert!(store.domains[WINDOW.0].arena.borrow().backed.len() <= 2);
    assert_eq!(store.occupied_rows(WINDOW), WINDOW_ROWS);
}

#[test]
fn freed_window_slab_index_is_reused_by_growth() {
    let Some(device) = cpu_device() else { return };
    let width = 32768;
    let store = windowed_store(device.clone(), 4096, 8, 4, 1024, width);
    let slab_rows = store.history_slab_rows(WINDOW);
    let arena = store.domains[WINDOW.0].arena.clone();
    // The window starts at row 0; another holder fills the rest of slab 0,
    // so decoding continues in a new slab 1.
    let mut state = commit(store.create().unwrap(), 8);
    assert_eq!(state.history_ranges(WINDOW), [(0, 8)]);
    let filler = take(&arena, 8, slab_rows - 8);
    for _ in 0..8 {
        store.provision(&state.demands(1), 0).unwrap();
        state = commit(state, 1);
    }
    let window_slabs = state
        .history_ranges(WINDOW)
        .iter()
        .map(|&(start, _)| start / slab_rows)
        .collect::<BTreeSet<_>>();
    assert_eq!(window_slabs, BTreeSet::from([1]));
    // Trimming released rows [0, 8); with the other holder gone, slab 0
    // holds no referenced row and the release rule frees it.
    drop(filler);
    assert_eq!(store.occupied_rows(WINDOW), 8);
    let charged = device.memory_usage().charged;
    assert!(store.shrink(ShrinkPolicy::Reclaim).unwrap() > 0);
    assert!(device.memory_usage().charged < charged);
    assert_eq!(arena.borrow().backed, BTreeSet::from([1]));
    assert_eq!(store.compactions(), Compactions::default());
    // Growth backs the lowest unbacked index, the freed slab 0, again.
    store
        .provision(
            &[RowDemand {
                domain: WINDOW,
                rows: store.free_rows(WINDOW) + 1,
            }],
            0,
        )
        .unwrap();
    assert_eq!(arena.borrow().backed, BTreeSet::from([0, 1]));
    let state = commit(state, 1);
    assert_eq!(store.occupied_rows(WINDOW), 8);
    assert_eq!(state.history_start(WINDOW), 9);
}

#[test]
fn compaction_repacks_one_domain() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device, 64, 64, 1, 128, 1);
    assert_eq!(store.span_bound(WINDOW), 17);
    store
        .provision(
            &[
                RowDemand {
                    domain: TOKEN,
                    rows: 64,
                },
                RowDemand {
                    domain: WINDOW,
                    rows: 128,
                },
            ],
            0,
        )
        .unwrap();
    // 17 accepted rows: contiguous in the Token domain, every other row in
    // the window domain.
    let token_arena = store.domains[TOKEN.0].arena.clone();
    let window_arena = store.domains[WINDOW.0].arena.clone();
    let mut state = store.create().unwrap();
    state.claims[TOKEN.0].append(take(&token_arena, 0, 17));
    let mut fillers = Vec::new();
    for row in 0..17 {
        state.claims[WINDOW.0].append(take(&window_arena, row * 2, 1));
        fillers.push(take(&window_arena, row * 2 + 1, 1));
    }
    state.position = 17;
    assert_eq!(state.compaction_needed(), Some(WINDOW));
    assert_eq!(
        state.compaction_demands(),
        [RowDemand {
            domain: WINDOW,
            rows: 17,
        }]
    );
    assert_eq!(state.span_count(), 17);
    let token_before = state.history_ranges(TOKEN);
    let OwnedCompactionPreparation::Ready(compaction) =
        OwnedCompaction::prepare(state, WINDOW, 64).ok().unwrap()
    else {
        panic!("a contiguous window run is free");
    };
    assert_eq!(compaction.domain(), WINDOW);
    // Only the window domain's planes are copied; they follow the Token
    // domain's two planes in the store's plane numbering.
    assert_eq!(
        compaction
            .copies()
            .iter()
            .map(|copy| copy.plane_index)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    let planes = compaction.bindings().history;
    assert!(compaction
        .copies()
        .iter()
        .all(|copy| planes[copy.plane_index].domain == WINDOW));
    let state = compaction.commit();
    assert_eq!(state.history_ranges(WINDOW).len(), 1);
    assert_eq!(state.history_ranges(TOKEN), token_before);
    assert_eq!(state.compaction_needed(), None);
    drop(fillers);
}

#[test]
fn shrink_and_census_cover_every_domain() {
    let Some(device) = cpu_device() else { return };
    let store = windowed_store(device.clone(), 8192, 16, 4, 8192, 4096);
    let window_slab = store.history_slab_rows(WINDOW);
    // Three window slabs: the growth claim is the two added.
    let demand = [RowDemand {
        domain: WINDOW,
        rows: 3 * window_slab,
    }];
    let slab_bytes = store.backing.borrow().history[WINDOW.0].slabs.slab_bytes();
    assert_eq!(
        store.growth_claim(&demand, 0).unwrap().minimum_bytes,
        2 * slab_bytes
    );
    store.provision(&demand, 0).unwrap();
    let state = commit(store.create().unwrap(), 40);
    let census = store
        .holding_census(&[Holder::State(&state)], &[], &[])
        .unwrap();
    assert_eq!(census.total(), store.committed_bytes());
    assert_eq!(census.total(), device.memory_usage().charged);
    assert_eq!(
        census.live,
        40 * store.history_row_bytes(TOKEN) + 16 * store.history_row_bytes(WINDOW)
    );
    let charged = device.memory_usage().charged;
    // Idle keeps each domain's occupied slabs and one spare.
    assert_eq!(store.shrink(ShrinkPolicy::Idle).unwrap(), slab_bytes);
    assert!(device.memory_usage().charged < charged);
    assert_eq!(store.domains[TOKEN.0].arena.borrow().backed.len(), 1);
    assert_eq!(store.domains[WINDOW.0].arena.borrow().backed.len(), 2);
    assert_eq!(
        store
            .holding_census(&[Holder::State(&state)], &[], &[])
            .unwrap()
            .total(),
        store.committed_bytes()
    );
}

#[test]
fn span_bounds_follow_each_domain_row_limit() {
    let plans = || {
        vec![
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Token {
                    components: vec![component(0, 1024)],
                },
                logical_rows: 262_144,
            },
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Window {
                    rows: 512,
                    components: vec![component(1, 1024)],
                },
                logical_rows: 4096,
            },
        ]
    };
    let (stored, shared) = domain::validate_domains(plans(), 262_144, 64).unwrap();
    assert!(shared.is_empty());
    // 8 KiB rows: 8192 rows per slab.
    assert_eq!(stored[0].slab_rows, 8192);
    assert_eq!(stored[0].span_bound, 262_144 / 8192 + 16);
    // A window history reaches n plus one advance: one slab of spans.
    assert_eq!(stored[1].span_bound, 1 + 16);
    assert_eq!(
        HistoryDomainKind::Window { rows: 512 }.row_limit(262_144, 64).unwrap(),
        576
    );
    assert_eq!(
        HistoryDomainKind::Window { rows: 512 }.row_limit(100, 64).unwrap(),
        100
    );
    // A window reservation below its row limit is refused.
    let mut short = plans();
    short[1].logical_rows = 575;
    assert!(domain::validate_domains(short, 262_144, 64).is_err());
}

#[test]
fn steady_footprints_charge_windows_independently_of_context() {
    let window = HistoryDomainLayout::Window {
        rows: 512,
        components: vec![component(1, 1024)],
    };
    let at = |context| window.steady_footprint(context, 64, 3, 2).unwrap();
    let rows = 3 * (512 + 64) + 2 * 512;
    assert_eq!(at(100_000).rows, rows as u64);
    assert_eq!(at(100_000), at(262_144));
    assert_eq!(at(100_000).rows_per_slab, 8192);
    assert_eq!(at(100_000).slabs, 1);
    let token = HistoryDomainLayout::Token {
        components: vec![component(0, 1024)],
    };
    let footprint = token.steady_footprint(100_000, 64, 3, 2).unwrap();
    assert_eq!(footprint.rows, 5 * 100_000);
    assert_eq!(footprint.slabs, (5 * 100_000u64).div_ceil(8192));
    let shared = HistoryDomainLayout::Shared {
        source: LayerRef::Target(0),
        layers: vec![LayerRef::Target(2)],
    };
    assert_eq!(
        shared.steady_footprint(100_000, 64, 3, 2).unwrap(),
        HistoryFootprint::default()
    );
    let block = HistoryDomainLayout::Block {
        rate: 4,
        components: vec![component(3, 8)],
    };
    assert!(matches!(
        block.steady_footprint(100_000, 64, 1, 0),
        Err(Error::UnsupportedHistoryDomain(HistoryDomainKind::Block { rate: 4 }))
    ));
}

#[test]
fn shared_layers_bind_their_source_regions() {
    let Some(device) = cpu_device() else { return };
    let store = StateStore::new(
        device,
        16,
        4,
        vec![
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Token {
                    components: vec![component(0, 4), component(1, 4)],
                },
                logical_rows: 16,
            },
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Shared {
                    source: LayerRef::Target(1),
                    layers: vec![LayerRef::Target(2), LayerRef::Target(3)],
                },
                logical_rows: 0,
            },
        ],
        vec![],
        banks(),
    )
    .unwrap();
    assert_eq!(store.history_domains().count(), 1);
    assert_eq!(store.sole_history_domain().unwrap(), TOKEN);
    assert_eq!(
        store.history_source(LayerRef::Target(3)),
        Some(HistorySource {
            domain: TOKEN,
            layer: LayerRef::Target(1),
        })
    );
    assert_eq!(
        store.history_source(LayerRef::Target(0)),
        Some(HistorySource {
            domain: TOKEN,
            layer: LayerRef::Target(0),
        })
    );
    assert_eq!(store.history_source(LayerRef::Target(4)), None);
    // Shared layers hold no storage.
    assert_eq!(store.history_planes().unwrap().len(), 4);
}

/// Qwen's configuration, one Token domain, is the one-domain store: the
/// same rows per slab, span bound, plane numbering, first slab and bytes.
#[test]
fn a_single_token_domain_is_the_one_domain_store() {
    let Some(device) = cpu_device() else { return };
    let components = vec![component(0, 4096), component(3, 4096)];
    let row_bytes = 2 * 2 * 4096 * 4;
    let store = StateStore::new(
        device.clone(),
        20_000,
        16,
        vec![HistoryDomainPlan {
            layout: HistoryDomainLayout::Token {
                components: components.clone(),
            },
            logical_rows: 40_000,
        }],
        vec![],
        banks(),
    )
    .unwrap();
    let slab_rows = history_rows_per_slab(row_bytes).unwrap();
    assert_eq!(store.history_row_bytes(TOKEN), row_bytes);
    assert_eq!(store.history_slab_rows(TOKEN), slab_rows);
    assert_eq!(
        store.span_bound(TOKEN),
        max_visible_spans(20_000, slab_rows).unwrap()
    );
    assert_eq!(store.committed_rows(TOKEN), slab_rows);
    let trace = store.allocation_trace().unwrap();
    assert_eq!(
        trace.history,
        [HistoryDomainTrace {
            kind: HistoryDomainKind::Token,
            capacity: 40_000,
            row_bytes,
            bytes: row_bytes * 40_000,
            slab_rows,
            span_bound: store.span_bound(TOKEN),
        }]
    );
    let planes = store.history_planes().unwrap();
    assert_eq!(
        planes
            .iter()
            .map(|plane| (plane.plane_index, plane.domain, plane.component_index, plane.layer))
            .collect::<Vec<_>>(),
        [
            (0, TOKEN, 0, LayerRef::Target(0)),
            (1, TOKEN, 0, LayerRef::Target(0)),
            (2, TOKEN, 1, LayerRef::Target(3)),
            (3, TOKEN, 1, LayerRef::Target(3)),
        ]
    );
    // Placement: the first history fills from row 0, the next starts mid
    // hole, and each grows in place.
    let first = commit(store.create().unwrap(), 10);
    let second = commit(store.create().unwrap(), 10);
    assert_eq!(first.history_ranges(TOKEN), [(0, 10)]);
    let middle = 10 + (slab_rows - 10 - 10) / 2;
    assert_eq!(second.history_ranges(TOKEN), [(middle, 10)]);
    let first = commit(first, 3);
    assert_eq!(first.history_ranges(TOKEN), [(0, 13)]);
    assert_eq!(first.history_start(TOKEN), 0);
    assert_eq!(store.committed_bytes(), device.memory_usage().charged);
}
