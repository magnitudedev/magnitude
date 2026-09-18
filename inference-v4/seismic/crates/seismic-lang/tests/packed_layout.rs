use seismic_lang::repr;

#[test]
fn snapshot_planes_cover_every_logical_prefix_with_row_independence() {
    for representation in repr::REPRS {
        let group = u64::from(representation.storage_group());
        for width in [0, 1, 17, 255, 256, 257] {
            let layout = representation.snapshot_layout(&[2, 3, width]).unwrap();
            assert_eq!(
                layout.strides,
                [3 * layout.physical_width, layout.physical_width, 1]
            );
            assert_eq!(layout.physical_width % group, 0);
            for (plane, expected) in layout.planes.iter().zip(representation.planes()) {
                assert_eq!(plane.plane, expected);
                assert_eq!(plane.elements, 6 * plane.elements_per_row);
                assert_eq!(
                    plane.elements_per_row,
                    expected.storage_elements(layout.physical_width).unwrap()
                );
                for prefix in 0..group {
                    let live = if width == 0 {
                        0
                    } else {
                        expected.storage_elements(prefix + width).unwrap()
                    };
                    assert!(
                        live <= plane.elements_per_row,
                        "{} width{width} prefix{prefix} {}",
                        representation.name,
                        expected.name
                    );
                }
            }
        }
        let empty = representation.snapshot_layout(&[0, 257]).unwrap();
        assert!(empty.planes.iter().all(|p| p.elements == 0));
        assert!(representation.snapshot_layout(&[]).is_none());
        assert!(representation.snapshot_layout(&[u64::MAX]).is_none());
        assert!(representation.snapshot_layout(&[u64::MAX, 257]).is_none());
    }
}
