use seismic_lang::types::DType;
use seismic_realization::dispatch::{GroupDispatch, TileDeclaration, TilePlacement};

#[test]
fn work_mapping_covers_each_logical_coordinate_once() {
    use seismic_realization::dispatch::WorkMapping;
    for outer in [1, 2, 3] {
        for middle in [1, 3] {
            for inner in [1, 2, 3, 4, 5, 6] {
                for step in [1, 2, 3] {
                    let mapping = WorkMapping::new(&[outer, middle, inner], &[1, 1, step]).unwrap();
                    let mut visited = std::collections::BTreeSet::new();
                    for item in 0..mapping.work_items() {
                        let base = mapping.coordinates(item).unwrap();
                        assert!(base[0] < outer && base[1] < middle);
                        for offset in 0..mapping.extents(item).unwrap()[2] {
                            assert!(base[2] + offset < inner);
                            assert!(visited.insert((base[0], base[1], base[2] + offset)));
                        }
                    }
                    let expected: std::collections::BTreeSet<_> = (0..outer)
                        .flat_map(|x| {
                            (0..middle).flat_map(move |y| (0..inner).map(move |z| (x, y, z)))
                        })
                        .collect();
                    assert_eq!(visited, expected);
                    assert!(mapping.coordinates(mapping.work_items()).is_err());
                }
            }
        }
    }
    let mapping = WorkMapping::new(&[2, 6], &[1, 3]).unwrap();
    assert_eq!(
        (0..4)
            .map(|i| mapping.coordinates(i).unwrap())
            .collect::<Vec<_>>(),
        [vec![0, 0], vec![0, 3], vec![1, 0], vec![1, 3]]
    );
}

#[test]
fn work_mapping_distinguishes_serial_empty_and_invalid_domains() {
    use seismic_realization::dispatch::WorkMapping;
    let serial = WorkMapping::new(&[], &[]).unwrap();
    assert_eq!(serial.work_items(), 1);
    assert_eq!(serial.coordinates(0).unwrap(), Vec::<u64>::new());
    for extents in [vec![0], vec![2, 0, 3], vec![u64::MAX, 0, u64::MAX]] {
        let mapping = WorkMapping::new(&extents, &vec![1; extents.len()]).unwrap();
        assert_eq!(mapping.work_items(), 0);
        assert!(mapping.coordinates(0).is_err());
    }
    assert!(WorkMapping::new(&[2], &[]).is_err());
    assert!(WorkMapping::new(&[2], &[0]).is_err());
    let tail = WorkMapping::new(&[3], &[2]).unwrap();
    assert_eq!(tail.work_items(), 2);
    assert_eq!(tail.coordinates(1).unwrap(), vec![2]);
    assert_eq!(tail.extents(0).unwrap()[0], 2);
    assert_eq!(tail.extents(1).unwrap()[0], 1);
    assert!(WorkMapping::new(&[u64::MAX, 2], &[1, 1]).is_err());
}
#[test]
fn declaration_units_and_padding_are_explicit() {
    let dispatch = GroupDispatch::new(7, 32, 4).unwrap();
    assert_eq!(dispatch.dispatched_lanes(), 256);
    assert_eq!(dispatch.participating_lanes(), 224);
    assert_eq!(dispatch.padding_lanes(), 32);
    let tile = |placement| TileDeclaration {
        symbol: "tile".into(),
        dtype: DType::BF16,
        capacity: 65,
        placement,
    };
    assert_eq!(
        tile(TilePlacement::Replicated).bytes(&dispatch).unwrap(),
        (130, 0)
    );
    assert_eq!(
        tile(TilePlacement::Distributed).bytes(&dispatch).unwrap(),
        (6, 0)
    );
    assert_eq!(
        tile(TilePlacement::GroupShared).bytes(&dispatch).unwrap(),
        (0, 520)
    );
    assert!(GroupDispatch::new(u64::MAX, 32, 4).is_err());
    assert!(GroupDispatch::new(1, 0, 4).is_err());
}

#[test]
fn physical_arrays_cover_logical_tiles_including_empty_and_partial_lanes() {
    for capacity in [0, 1, 31, 32, 33, 65] {
        for dtype in [DType::F16, DType::F32] {
            let dispatch = GroupDispatch::new(7, 32, 4).unwrap();
            for placement in [
                TilePlacement::Replicated,
                TilePlacement::Distributed,
                TilePlacement::GroupShared,
                TilePlacement::GroupWide,
            ] {
                let tile = TileDeclaration {
                    symbol: "t".into(),
                    dtype,
                    capacity,
                    placement: placement.clone(),
                };
                let layout = tile.layout(&dispatch).unwrap();
                let width = u64::from(dtype.bytes());
                match placement {
                    TilePlacement::Replicated => {
                        assert_eq!(layout.private_elements_per_lane, capacity.max(1));
                        assert_eq!(layout.shared_elements_per_item, 0);
                    }
                    TilePlacement::Distributed => {
                        let slots = layout.private_elements_per_lane;
                        assert!(slots > 0);
                        assert!(slots * 32 >= capacity);
                        assert!(slots == 1 || (slots - 1) * 32 < capacity);
                        assert_eq!(layout.shared_elements_per_item, 0);
                    }
                    TilePlacement::GroupShared | TilePlacement::GroupWide => {
                        assert_eq!(layout.private_elements_per_lane, 0);
                        assert_eq!(layout.shared_elements_per_item, capacity.max(1));
                    }
                }
                assert_eq!(
                    layout.private_bytes_per_lane,
                    layout.private_elements_per_lane * width
                );
                // A group-wide array exists once per group; an item-owned one once per item.
                let items = if placement == TilePlacement::GroupWide { 1 } else { 4 };
                assert_eq!(
                    layout.shared_bytes_per_group,
                    layout.shared_elements_per_item * width * items
                );
                assert_eq!(
                    tile.bytes(&dispatch).unwrap(),
                    (layout.private_bytes_per_lane, layout.shared_bytes_per_group)
                );
            }
        }
    }
}

#[test]
fn storage_rejects_invalid_widths_and_byte_overflow() {
    let mut dispatch = GroupDispatch::new(1, 32, 4).unwrap();
    let mut tile = TileDeclaration {
        symbol: "t".into(),
        dtype: DType::F32,
        capacity: u64::MAX,
        placement: TilePlacement::Replicated,
    };
    assert!(tile.layout(&dispatch).is_err());
    tile.placement = TilePlacement::GroupShared;
    // Per-item bytes fit, but per-group bytes do not.
    tile.capacity = u64::MAX / 4;
    assert!(tile.layout(&dispatch).is_err());
    tile.capacity = 1;
    dispatch.lanes_per_item = 0;
    assert!(tile.layout(&dispatch).is_err());
    dispatch.lanes_per_item = 32;
    dispatch.items_per_group = 0;
    assert!(tile.layout(&dispatch).is_err());
}
