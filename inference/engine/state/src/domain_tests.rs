//! History domains in one store: window trimming and sharing, per-domain
//! placement, compaction and accounting, and the single Token domain's
//! identity with the one-domain store.

use super::*;
use seismic::BackendName;

const TOKEN: HistoryDomainId = HistoryDomainId(0);
const WINDOW: HistoryDomainId = HistoryDomainId(1);

fn cpu_device() -> Rc<Device> {
    let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
    Rc::new(catalog.open_backend(BackendName::Cpu).expect("the CPU backend opens"))
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
) -> StoreBindings {
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
    .1
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
    let device = cpu_device();
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
    // Freed rows stay in the history's page (the domain's only page): no
    // other history takes them, so no whole page is free.
    assert_eq!(store.free_rows(WINDOW), 0);
    // The history itself keeps growing in place into them.
    let state = commit(state, 50);
    assert_eq!(state.history_ranges(WINDOW).len(), 1);
}

#[test]
fn windows_smaller_than_an_advance_write_every_row_then_trim() {
    let device = cpu_device();
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
    let device = cpu_device();
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
    let device = cpu_device();
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
    let device = cpu_device();
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
    let device = cpu_device();
    const WINDOW_ROWS: usize = 8;
    // 2 planes of 32768 F32 values: 256 KiB rows, 256 rows per slab.
    let width = 32768;
    let mut bindings = windowed_store(device.clone(), 4096, WINDOW_ROWS, 4, 512, width);
    let store = Rc::clone(&bindings);
    let slab_rows = store.history_slab_rows(WINDOW);
    assert_eq!(slab_rows, SLAB_ROW_TILE);
    // Eight rows plus a four-row advance fit one page: at most two spans.
    assert_eq!(store.history_page_rows(WINDOW), SLAB_ROW_TILE);
    assert_eq!(store.span_limit(WINDOW), 2);
    // Back both slabs, so the window runs from slab 0 straight into slab 1.
    bindings
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
    let step = |bindings: &mut StoreBindings, state: SequenceState, rows: usize| {
        bindings.provision(&state.demands(rows), 0).unwrap();
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
        assert!(ranges.len() <= store.span_limit(WINDOW));
        let start = state.history_start(WINDOW);
        assert_eq!(start, state.position().saturating_sub(WINDOW_ROWS));
        for (offset, row) in rows_of(&ranges).into_iter().enumerate() {
            assert_eq!(read(&store, row), start + offset, "row {row}");
        }
    };
    let mut state = store.create().unwrap();
    let mut crossed = false;
    for _ in 0..700 {
        state = step(&mut bindings, state, 1);
        check(&state);
        crossed |= state.history_ranges(WINDOW).len() > 1;
        // Idle release keeps one spare slab and never copies.
        bindings
            .shrink_with(ShrinkPolicy::Idle, |_, _| -> Result<(), Error> {
                panic!("idle release never copies")
            })
            .unwrap();
    }
    assert!(crossed, "the window spanned a slab edge");
    assert_eq!(store.occupied_rows(WINDOW), WINDOW_ROWS);

    // Reclaim frees whichever slab the window does not occupy; the next
    // growth adds the lowest unbacked index again.
    let before = device.device_ledger().charged;
    let released = bindings.shrink(ShrinkPolicy::Reclaim).unwrap();
    let occupied_slabs = state
        .history_ranges(WINDOW)
        .iter()
        .map(|&(start, _)| start / slab_rows)
        .collect::<BTreeSet<_>>();
    let backed = store.domains[WINDOW.0].arena.borrow().backed.clone();
    assert!(backed.len() <= 2);
    assert!(occupied_slabs.is_subset(&backed));
    if released > 0 {
        assert!(device.device_ledger().charged < before);
    }
    check(&state);
    for _ in 0..600 {
        state = step(&mut bindings, state, 3);
        check(&state);
    }
    assert!(store.domains[WINDOW.0].arena.borrow().backed.len() <= 2);
    assert_eq!(store.occupied_rows(WINDOW), WINDOW_ROWS);
}

#[test]
fn freed_window_slab_index_is_reused_by_growth() {
    let device = cpu_device();
    let width = 32768;
    let mut store = windowed_store(device.clone(), 4096, 8, 4, 1024, width);
    let slab_rows = store.history_slab_rows(WINDOW);
    let arena = store.domains[WINDOW.0].arena.clone();
    // The window starts at row 0 and decodes through slab 0 (one page)
    // into slab 1.
    let mut state = commit(store.create().unwrap(), 8);
    assert_eq!(state.history_ranges(WINDOW), [(0, 8)]);
    for _ in 0..slab_rows {
        store.provision(&state.demands(1), 0).unwrap();
        state = commit(state, 1);
    }
    let window_slabs = state
        .history_ranges(WINDOW)
        .iter()
        .map(|&(start, _)| start / slab_rows)
        .collect::<BTreeSet<_>>();
    assert_eq!(window_slabs, BTreeSet::from([1]));
    // Trimming released every row of slab 0, so it holds no referenced row
    // and the release rule frees it.
    assert_eq!(store.occupied_rows(WINDOW), 8);
    let charged = device.device_ledger().charged;
    assert!(store.shrink(ShrinkPolicy::Reclaim).unwrap() > 0);
    assert!(device.device_ledger().charged < charged);
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
    assert_eq!(state.history_start(WINDOW), slab_rows + 1);
}

/// A relocation moves one domain's last page: only that domain is blocked,
/// only its slab regions are copied, and the other domain keeps its rows.
#[test]
fn relocation_moves_one_domain() {
    let device = cpu_device();
    let mut store = windowed_store(device, 1024, 64, 1, 1024, 1);
    store
        .provision(
            &[
                RowDemand {
                    domain: TOKEN,
                    rows: 1024,
                },
                RowDemand {
                    domain: WINDOW,
                    rows: 1024,
                },
            ],
            0,
        )
        .unwrap();
    let checkpoint = commit(store.create().unwrap(), 10).checkpoint();
    let state = checkpoint.fork();
    // Another history takes the window row after this history's end.
    let window_end = rows_of(&state.history_ranges(WINDOW))[9] + 1;
    let window_arena = store.domains[WINDOW.0].arena.clone();
    let filler = take(&window_arena, window_end, 1);
    assert_eq!(state.blocked_tails(), [WINDOW]);
    let token_before = state.history_ranges(TOKEN);
    let relocation = OwnedTailRelocation::prepare(state, WINDOW).ok().unwrap();
    assert_eq!(relocation.domain(), WINDOW);
    // The copy covers the window domain's two planes (key and value) only.
    let copy = relocation.copy();
    assert_eq!(copy.planes().len(), 2);
    assert_eq!(copy.copies().len(), 2);
    assert_eq!(copy.rows(), 10);
    let state = relocation.commit();
    assert_eq!(state.history_ranges(WINDOW).len(), 1);
    assert_eq!(state.history_ranges(TOKEN), token_before);
    assert!(state.blocked_tails().is_empty());
    let state = commit(state, 1);
    assert_eq!(state.history_ranges(WINDOW).len(), 1);
    drop(filler);
}

/// Two live forks of one checkpoint end at the same row of a partial page,
/// whose next rows either could take first: each is blocked while the other
/// ends there, the checkpoint blocks neither, and once one relocates the
/// other grows in place. (Regression #237.)
#[test]
fn sibling_forks_ending_together_relocate_all_but_one() {
    let device = cpu_device();
    let mut store = windowed_store(device, 1024, 64, 1, 1024, 1);
    store
        .provision(
            &[
                RowDemand {
                    domain: TOKEN,
                    rows: 1024,
                },
                RowDemand {
                    domain: WINDOW,
                    rows: 1024,
                },
            ],
            0,
        )
        .unwrap();
    let checkpoint = commit(store.create().unwrap(), 10).checkpoint();
    let only = checkpoint.fork();
    assert!(only.blocked_tails().is_empty());
    let sibling = checkpoint.fork();
    assert_eq!(only.blocked_tails(), [TOKEN, WINDOW]);
    assert_eq!(sibling.blocked_tails(), [TOKEN, WINDOW]);
    assert!(OwnedStateAdvance::begin(sibling, 1).is_err());
    let mut sibling = checkpoint.fork();
    for domain in [TOKEN, WINDOW] {
        sibling = OwnedTailRelocation::prepare(sibling, domain)
            .ok()
            .unwrap()
            .commit();
    }
    assert!(sibling.blocked_tails().is_empty());
    assert!(only.blocked_tails().is_empty());
    let end = rows_of(&only.history_ranges(TOKEN))[9] + 1;
    let only = commit(only, 1);
    assert_eq!(
        rows_of(&only.history_ranges(TOKEN))[10],
        end,
        "the last sibling grows in place"
    );
    let sibling = commit(sibling, 1);
    drop((only, sibling, checkpoint));
}
#[test]
fn shrink_and_census_cover_every_domain() {
    let device = cpu_device();
    let mut store = windowed_store(device.clone(), 8192, 16, 4, 8192, 4096);
    let window_slab = store.history_slab_rows(WINDOW);
    // Three window slabs: the growth claim is the two added.
    let demand = [RowDemand {
        domain: WINDOW,
        rows: 3 * window_slab,
    }];
    let slab_bytes = store.backing.borrow().history[WINDOW.0].slabs.slab_bytes();
    assert_eq!(
        store.growth_claim(&demand, &[], 0).unwrap().minimum_bytes,
        2 * slab_bytes
    );
    store.provision(&demand, 0).unwrap();
    let state = commit(store.create().unwrap(), 40);
    let census = store
        .holding_census(&[Holder::State(&state)], &[], &[])
        .unwrap();
    assert_eq!(census.total(), store.committed_bytes());
    assert_eq!(census.total(), device.device_ledger().charged);
    assert_eq!(
        census.live,
        40 * store.history_row_bytes(TOKEN) + 16 * store.history_row_bytes(WINDOW)
    );
    let charged = device.device_ledger().charged;
    // Idle keeps each domain's occupied slabs and one spare.
    assert_eq!(store.shrink(ShrinkPolicy::Idle).unwrap(), slab_bytes);
    assert!(device.device_ledger().charged < charged);
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
fn span_limits_follow_each_domain_row_limit() {
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
    // 8 KiB rows: the smallest page keeping 262,144 rows within 63 spans is
    // 4,352 rows (17 tiles: 61 pages plus a partial first), and a 64 MiB
    // slab holds one such page.
    assert_eq!(stored[0].geometry.page_rows, 4352);
    assert_eq!(stored[0].geometry.slab_rows, 4352);
    assert_eq!(stored[0].geometry.span_limit, 62);
    // A window history holds n plus one advance: 576 rows in 256-row pages.
    assert_eq!(stored[1].geometry.page_rows, 256);
    assert_eq!(stored[1].geometry.slab_rows, 8192);
    assert_eq!(stored[1].geometry.span_limit, 4);
    // A Shared reader of the Token domain also sees an advance's appended
    // rows, so its page grows to keep the reader within the limit.
    let mut with_reader = plans();
    with_reader.push(HistoryDomainPlan {
        layout: HistoryDomainLayout::Shared {
            source: LayerRef::Target(0),
            layers: vec![LayerRef::Target(2)],
        },
        logical_rows: 0,
    });
    let (stored, shared) = domain::validate_domains(with_reader, 262_144, 64).unwrap();
    assert_eq!(shared.len(), 1);
    assert_eq!(
        stored[0].geometry,
        history_geometry(8192, 262_144, 64).unwrap()
    );
    assert_eq!(stored[0].geometry.page_rows, 4608);
    assert!(stored[0].geometry.span_limit <= MAX_HISTORY_SPANS);
    assert_eq!(stored[1].geometry.span_limit, 4);
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
fn shared_layers_bind_their_source_regions() {
    let device = cpu_device();
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
    .unwrap()
    .0;
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
    let device = cpu_device();
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
    .unwrap()
    .0;
    let geometry = history_geometry(row_bytes, 20_000, 0).unwrap();
    let slab_rows = geometry.slab_rows;
    assert_eq!(store.history_row_bytes(TOKEN), row_bytes);
    assert_eq!(store.history_slab_rows(TOKEN), slab_rows);
    assert_eq!(store.history_page_rows(TOKEN), geometry.page_rows);
    assert_eq!(store.span_limit(TOKEN), geometry.span_limit);
    assert_eq!(store.committed_rows(TOKEN), slab_rows);
    let trace = store.allocation_trace().unwrap();
    assert_eq!(
        trace.history,
        [HistoryDomainTrace {
            kind: HistoryDomainKind::Token,
            capacity: geometry.reserved_rows(40_000).unwrap(),
            row_bytes,
            bytes: row_bytes * geometry.reserved_rows(40_000).unwrap() as u64,
            slab_rows,
            page_rows: geometry.page_rows,
            span_limit: geometry.span_limit,
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
    // Placement: the first history fills from row 0, the next takes the
    // middle page of the free pages after it, and each grows in place.
    let first = commit(store.create().unwrap(), 10);
    let second = commit(store.create().unwrap(), 10);
    assert_eq!(first.history_ranges(TOKEN), [(0, 10)]);
    let free_pages = slab_rows / geometry.page_rows - 1;
    let middle = geometry.page_rows * (1 + (free_pages - 1) / 2);
    assert_eq!(second.history_ranges(TOKEN), [(middle, 10)]);
    let first = commit(first, 3);
    assert_eq!(first.history_ranges(TOKEN), [(0, 13)]);
    assert_eq!(first.history_start(TOKEN), 0);
    assert_eq!(store.committed_bytes(), device.device_ledger().charged);
}

/// A Token domain and one selected-attention indexer of blocks of four: its
/// raw keys in a Window(3) domain and its block keys in a Block(4) domain.
const RAW: HistoryDomainId = HistoryDomainId(1);
const BLOCKS: HistoryDomainId = HistoryDomainId(2);
const INDEX_KEYS: LayerRef = LayerRef::IndexKeys { block: 0, part: 0 };

fn indexed_store(device: Rc<Device>, context: usize, max_advance: usize) -> StoreBindings {
    let keys = |coordinates| {
        ComponentDescriptor::produced(INDEX_KEYS, DType::F32, 4, coordinates).unwrap()
    };
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
                    rows: 3,
                    components: vec![keys(true)],
                },
                logical_rows: context,
            },
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Block {
                    rate: 4,
                    components: vec![keys(false)],
                },
                logical_rows: context / 4,
            },
        ],
        vec![],
        banks(),
    )
    .unwrap()
    .1
}

#[test]
fn block_histories_hold_the_complete_blocks() {
    let device = cpu_device();
    let store = indexed_store(device, 64, 16);
    // The indexer's domains follow the one attention reads; a launch row
    // reads each.
    assert_eq!(store.history_reads().len(), 3);
    assert_eq!(store.index_key_domains(0, 0), Some((Some(RAW), BLOCKS)));
    assert!(!store.is_produced_domain(HistoryDomainId(0)));
    assert!(store.is_produced_domain(RAW) && store.is_produced_domain(BLOCKS));
    // Raw keys carry their coordinates; block keys do not.
    assert_eq!(store.history_components(RAW)[0].planes().len(), 2);
    assert_eq!(store.history_components(BLOCKS)[0].planes().len(), 1);
    assert_eq!(store.index_key_domains(1, 0), None);
    // Ten rows from 0 complete blocks 0 and 1: two block rows, the
    // advance's destinations in block order.
    let advance = OwnedStateAdvance::begin(store.create().unwrap(), 10).ok().unwrap();
    assert_eq!(advance.bindings().destinations[BLOCKS.0].len(), 2);
    assert_eq!(advance.bindings().destinations[RAW.0].len(), 10);
    let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
        panic!("an advance commits");
    };
    assert_eq!(rows_of(&state.history_ranges(BLOCKS)).len(), 2);
    assert_eq!(state.history_start(BLOCKS), 0);
    // The raw keys of the incomplete block are among the last three rows.
    assert_eq!(state.history_start(RAW), 7);
    assert_eq!(store.occupied_rows(BLOCKS), 2);
    // Six more rows from 10 complete blocks 2 and 3 (positions 11 and 15).
    let state = commit(state, 6);
    assert_eq!(rows_of(&state.history_ranges(BLOCKS)).len(), 4);
    assert_eq!(store.occupied_rows(BLOCKS), 4);
    assert_eq!(store.occupied_rows(RAW), 3);
}

#[test]
fn rollback_keeps_the_blocks_of_the_accepted_prefix() {
    let device = cpu_device();
    let store = indexed_store(device, 64, 16);
    let state = commit(store.create().unwrap(), 6);
    let before = state.history_ranges(BLOCKS);
    assert_eq!(rows_of(&before).len(), 1);
    // A verification of eight rows from 6 would complete blocks 1 and 2
    // (positions 7 and 11); accepting three rows (positions 6..9) keeps
    // block 1 only.
    let advance = OwnedStateAdvance::begin(state, 8).ok().unwrap();
    let completed = advance.bindings().destinations[BLOCKS.0].clone();
    assert_eq!(completed.len(), 2);
    let OwnedAdvanceResolution::Committed(state) = advance.commit(3).ok().unwrap() else {
        panic!("a prefix commits");
    };
    assert_eq!(
        rows_of(&state.history_ranges(BLOCKS)),
        rows_of(&before).into_iter().chain([completed[0]]).collect::<Vec<_>>()
    );
    assert_eq!(store.occupied_rows(BLOCKS), 2);
    // Accepting one row (position 6) completes no block.
    let advance = OwnedStateAdvance::begin(state, 4).ok().unwrap();
    assert_eq!(advance.bindings().destinations[BLOCKS.0].len(), 1);
    let OwnedAdvanceResolution::Committed(state) = advance.commit(1).ok().unwrap() else {
        panic!("a prefix commits");
    };
    assert_eq!(state.position(), 10);
    assert_eq!(rows_of(&state.history_ranges(BLOCKS)).len(), 2);
    assert_eq!(store.occupied_rows(BLOCKS), 2);
}

#[test]
fn checkpoints_and_successors_share_the_block_rows() {
    let device = cpu_device();
    let store = indexed_store(device, 64, 16);
    let state = commit(store.create().unwrap(), 9);
    // A checkpoint at 9 references blocks [0, 2).
    let checkpoint = state.checkpoint();
    assert_eq!(checkpoint.history_ranges(BLOCKS), state.history_ranges(BLOCKS));
    drop(state);
    assert_eq!(store.occupied_rows(BLOCKS), 2);
    // A fork resumed from it continues the same blocks: three rows from 9
    // complete block 2 (position 11).
    let fork = commit(checkpoint.fork(), 3);
    assert_eq!(rows_of(&fork.history_ranges(BLOCKS))[..2], rows_of(&checkpoint.history_ranges(BLOCKS))[..]);
    assert_eq!(store.occupied_rows(BLOCKS), 3);
    // A successor of an in-flight advance reads the blocks its predecessor
    // completes: one row from 12 after four from 12..
    let first = OwnedStateAdvance::begin(fork, 4).ok().unwrap();
    let completed = first.bindings().destinations[BLOCKS.0].clone();
    assert_eq!(completed.len(), 1);
    let second = first.successor(4).unwrap();
    assert_eq!(rows_of(&second.history_ranges(BLOCKS)).last(), completed.last());
    assert_eq!(second.bindings().destinations[BLOCKS.0].len(), 1);
    let OwnedAdvanceResolution::Committed(state) = first.commit_all().ok().unwrap() else {
        panic!("an advance commits");
    };
    let second = second.attach(state).ok().unwrap();
    let OwnedAdvanceResolution::Committed(state) = second.commit_all().ok().unwrap() else {
        panic!("an advance commits");
    };
    assert_eq!(state.position(), 20);
    assert_eq!(rows_of(&state.history_ranges(BLOCKS)).len(), 5);
    drop(checkpoint);
    assert_eq!(store.occupied_rows(BLOCKS), 5);
}

/// V4.1's compressed layer at rate 2 in one store: its 128-row window
/// (domain 0), its pooled source's incomplete group, a Window(1) of F32
/// projections and gate scores with coordinates (domain 1), and its rows
/// beside their indexer keys, a Block(2) (domain 2).
const PENDING: HistoryDomainId = HistoryDomainId(1);
const POOLED: HistoryDomainId = HistoryDomainId(2);

fn pooled_store(device: Rc<Device>, context: usize, max_advance: usize) -> StoreBindings {
    let source = LayerRef::Pooled { block: 0, part: 1 };
    let keys = LayerRef::IndexKeys { block: 0, part: 1 };
    StateStore::new(
        device,
        context,
        max_advance,
        vec![
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Window {
                    rows: 128,
                    components: vec![component(0, 4)],
                },
                logical_rows: context,
            },
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Window {
                    rows: 1,
                    components: vec![
                        ComponentDescriptor::produced(source, DType::F32, 8, true).unwrap(),
                    ],
                },
                logical_rows: context,
            },
            HistoryDomainPlan {
                layout: HistoryDomainLayout::Block {
                    rate: 2,
                    components: vec![
                        ComponentDescriptor::model_history(
                            source,
                            CodecSpec::stored_key(DType::BF16, 4),
                            1,
                        )
                        .unwrap(),
                        ComponentDescriptor::produced(keys, DType::BF16, 4, false).unwrap(),
                    ],
                },
                logical_rows: context / 2,
            },
        ],
        vec![],
        banks(),
    )
    .unwrap()
    .1
}

/// What a pooled history references at its position: its pending group's
/// first position and row count, and its complete groups' row count.
fn pooled_extent(state: &SequenceState) -> (usize, usize, usize) {
    (
        state.history_start(PENDING),
        rows_of(&state.history_ranges(PENDING)).len(),
        rows_of(&state.history_ranges(POOLED)).len(),
    )
}

#[test]
fn a_rejected_verify_keeps_the_accepted_prefix_partial_group() {
    let device = cpu_device();
    let store = pooled_store(device.clone(), 64, 8);
    assert_eq!(store.pooled_domains(0, 1), Some((Some(PENDING), POOLED)));
    assert_eq!(store.index_key_domains(0, 1), Some((None, POOLED)));
    // Three rows (positions 0..3): group 0 complete, position 2 pending.
    let state = commit(store.create().unwrap(), 3);
    assert_eq!(pooled_extent(&state), (2, 1, 1));
    // A verify of three rows from 3 completes groups 1 (position 3) and 2
    // (position 5); accepting two (positions 3, 4) keeps group 1 and leaves
    // position 4 pending.
    let advance = OwnedStateAdvance::begin(state, 3).ok().unwrap();
    assert_eq!(advance.bindings().destinations[POOLED.0].len(), 2);
    let OwnedAdvanceResolution::Committed(state) = advance.commit(2).ok().unwrap() else {
        panic!("a prefix commits");
    };
    assert_eq!(state.position(), 5);
    // A cold advance of the five accepted rows references the same rows.
    let cold_store = pooled_store(device, 64, 8);
    let cold = commit(cold_store.create().unwrap(), 5);
    assert_eq!(pooled_extent(&state), pooled_extent(&cold));
    assert_eq!(pooled_extent(&state), (4, 1, 2));
    // The rejected group's row is released.
    assert_eq!(store.occupied_rows(POOLED), cold_store.occupied_rows(POOLED));
    assert_eq!(store.occupied_rows(PENDING), cold_store.occupied_rows(PENDING));
    // Decoding on from the partial group completes group 2 at position 5.
    let state = commit(state, 1);
    assert_eq!(pooled_extent(&state), (5, 1, 3));
}

#[test]
fn produced_domains_follow_the_per_token_ones() {
    let device = cpu_device();
    let keys = || ComponentDescriptor::produced(INDEX_KEYS, DType::F32, 4, false).unwrap();
    let pooled = LayerRef::Pooled { block: 0, part: 1 };
    let rows = || {
        ComponentDescriptor::model_history(pooled, CodecSpec::stored_key(DType::BF16, 4), 1)
            .unwrap()
    };
    let plan = |layout| HistoryDomainPlan {
        logical_rows: 64,
        layout,
    };
    let token = |components| plan(HistoryDomainLayout::Token { components });
    let block = |components| plan(HistoryDomainLayout::Block { rate: 4, components });
    let store = |plans| StateStore::new(device.clone(), 64, 16, plans, vec![], banks());
    // A produced domain before a per-token one.
    assert!(store(vec![block(vec![keys()]), token(vec![component(0, 1)])]).is_err());
    // A per-token domain holding produced rows, or a Block holding
    // per-token rows.
    assert!(store(vec![token(vec![component(0, 1), keys()])]).is_err());
    assert!(store(vec![token(vec![component(0, 1)]), block(vec![component(1, 1)])]).is_err());
    // A pooled source's rows beside their indexer's keys, read through its
    // attended rows.
    let (_, bindings) = store(vec![token(vec![component(0, 1)]), block(vec![rows(), keys()])])
        .unwrap();
    assert_eq!(
        bindings.history_source(pooled).map(|source| source.domain),
        Some(HistoryDomainId(1))
    );
    assert_eq!(bindings.index_key_domains(0, 0), Some((None, HistoryDomainId(1))));
    assert_eq!(bindings.history_reads().read(pooled), Some(1));
    assert_eq!(bindings.history_reads().read(INDEX_KEYS), None);
    assert!(ComponentDescriptor::produced(LayerRef::Target(0), DType::F32, 4, true).is_err());
    assert!(ComponentDescriptor::new(INDEX_KEYS, CodecSpec::dense(DType::F32, 4, 4), 1).is_err());
}

/// A batch's most rows in a domain are its rows at the worst position: one
/// per row, or every block the rows can complete.
#[test]
fn a_batchs_most_rows_are_its_worst_positions() {
    let kinds = [
        HistoryDomainKind::Token,
        HistoryDomainKind::Window { rows: 3 },
        HistoryDomainKind::Block { rate: 1 },
        HistoryDomainKind::Block { rate: 4 },
        HistoryDomainKind::Block { rate: 7 },
    ];
    for kind in kinds {
        for count in 0..20 {
            let worst = (0..40).map(|position| kind.advance_rows(position, count)).max().unwrap();
            assert_eq!(kind.most_advance_rows(count), worst, "{kind:?} at {count} rows");
        }
    }
}

/// The backing a submission batch newly takes in each domain, at its worst
/// page offset (a history whose last page holds none of it), is its
/// `batch_state_bytes`: a fresh history for the Token and Window domains, and
/// for the Block(4) domain one at position 3, whose batch completes its
/// first block.
#[test]
fn a_batch_takes_its_state_bytes_of_pages() {
    let keys = |coordinates| ComponentDescriptor::produced(INDEX_KEYS, DType::F32, 4, coordinates).unwrap();
    let layouts = [
        HistoryDomainLayout::Token { components: vec![component(0, 1)] },
        HistoryDomainLayout::Window { rows: 3, components: vec![keys(true)] },
        HistoryDomainLayout::Block { rate: 4, components: vec![keys(false)] },
    ];
    let (context, max_advance) = (64, 16);
    for rows in 1..=4 {
        for (index, (layout, at)) in layouts.iter().zip([0, 0, 3]).enumerate() {
            let domain = HistoryDomainId(index);
            let store = indexed_store(cpu_device(), context, max_advance);
            let state = match at {
                0 => store.create().unwrap(),
                at => commit(store.create().unwrap(), at),
            };
            let free = store.free_rows(domain);
            let advance = OwnedStateAdvance::begin(state, rows).ok().unwrap();
            let taken = (free - store.free_rows(domain)) as u64 * store.history_row_bytes(domain);
            let single = std::slice::from_ref(layout);
            assert_eq!(
                taken,
                batch_state_bytes(single, context, max_advance, rows as u64).unwrap(),
                "{:?} at {rows} rows",
                layout.kind()
            );
            let _ = advance.abort();
        }
        // The store's every domain at once.
        let each = layouts
            .iter()
            .map(|layout| batch_state_bytes(std::slice::from_ref(layout), context, max_advance, rows as u64).unwrap())
            .sum::<u64>();
        assert_eq!(batch_state_bytes(&layouts, context, max_advance, rows as u64).unwrap(), each);
    }
}

/// A window history occupies at most its row limit's pages and the partial
/// page at its front however long it grows, so `batch_state_bytes` of its
/// rows from a page boundary caps there: as a sequence advances far past
/// its window, the pages the store holds for it never exceed that figure,
/// and reach it once the window straddles a page boundary in flight.
#[test]
fn a_window_historys_pages_stay_within_its_capped_state_bytes() {
    let (context, window, max_advance) = (4096, 300, 16);
    let mut bindings = windowed_store(cpu_device(), context, window, max_advance, 1024, 2);
    let store = Rc::clone(&bindings);
    let single = [HistoryDomainLayout::Window { rows: window, components: vec![component(1, 2)] }];
    let row_bytes = store.history_row_bytes(WINDOW);
    let page = store.history_page_rows(WINDOW);
    let cap = window_pages(page, window + max_advance) * page;
    let free = store.free_rows(WINDOW);
    let mut state = store.create().unwrap();
    let mut most = 0;
    for _ in 0..context / max_advance {
        bindings.provision(&state.demands(max_advance), 0).unwrap();
        let advance = OwnedStateAdvance::begin(state, max_advance).ok().unwrap();
        let through = advance.position() + max_advance;
        let held = free - store.free_rows(WINDOW);
        let bytes = batch_state_bytes(&single, context, max_advance, through as u64).unwrap();
        assert!(held as u64 * row_bytes <= bytes, "{held} rows held through {through}");
        most = most.max(held);
        let OwnedAdvanceResolution::Committed(next) = advance.commit_all().ok().unwrap() else {
            panic!("an attention advance commits");
        };
        state = next;
    }
    assert_eq!(most, cap, "a window straddling a page boundary in flight holds the capped pages");
    assert_eq!(batch_state_bytes(&single, context, max_advance, context as u64).unwrap(), cap as u64 * row_bytes);
}
