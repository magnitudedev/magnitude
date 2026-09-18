use seismic_engine::{
    inputs::{
        media::{DType, PreparedMedia, PreparedTensor},
        TokenId,
    },
    models::qwen35::preparation::{interpret, ImageGeometry, InputPlan},
};
fn geometry() -> ImageGeometry {
    ImageGeometry {
        channels: 1,
        temporal_patch: 1,
        patch: 1,
        merge: 2,
        image_token: TokenId(99),
        start_token: TokenId(98),
        end_token: TokenId(100),
    }
}
fn media(grids: &[[i64; 3]], values: Vec<f32>) -> PreparedMedia {
    PreparedMedia::new(
        "a".repeat(64),
        vec![
            PreparedTensor::new(
                "pixel_values".into(),
                DType::F32,
                vec![values.len(), 1],
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            )
            .unwrap(),
            PreparedTensor::new(
                "image_grid_thw".into(),
                DType::I64,
                vec![grids.len(), 3],
                grids
                    .iter()
                    .flatten()
                    .flat_map(|n| n.to_le_bytes())
                    .collect(),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}
fn tokens(ids: &[u32]) -> Vec<TokenId> {
    ids.iter().copied().map(TokenId).collect()
}
#[test]
fn v3_image_coordinates_continuation_and_history_are_preserved() {
    let prepared = interpret(
        tokens(&[7, 98, 99, 99, 100, 8]),
        &media(&[[1, 2, 4]], (0..8).map(|n| n as f32).collect()),
        &"a".repeat(64),
        &geometry(),
    )
    .unwrap();
    assert_eq!(
        prepared.plan.rotary(0, 6).unwrap(),
        vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [4, 4, 4],
            [5, 5, 5]
        ]
    );
    assert_eq!(prepared.plan.rotary(6, 2).unwrap(), vec![[6; 3], [7; 3]]);
    let square = interpret(
        tokens(&[7, 98, 99, 99, 99, 99, 100, 8]),
        &media(&[[1, 4, 4]], vec![0.; 16]),
        &"a".repeat(64),
        &geometry(),
    )
    .unwrap();
    assert_eq!(square.plan.continuation(), 6);
    assert_eq!(square.plan.rotary(8, 1).unwrap(), vec![[6; 3]]);
    assert!(square.plan.layout().boundary(4));
    assert!(!square.plan.layout().language(4).unwrap());
    assert!(square.plan.layout().language(8).unwrap());
    let text = InputPlan::text(tokens(&[1, 2])).unwrap();
    assert_eq!(text.rotary(1, 3).unwrap(), vec![[1; 3], [2; 3], [3; 3]]);
    assert!(text.rotary(i32::MAX as usize, 1).is_err());
    assert!(text.rotary(usize::MAX, 1).is_err());
}
#[test]
fn exact_placeholder_processor_finite_pixels_and_grid_extents_are_required() {
    let pixels = media(&[[1, 2, 4]], vec![0.; 8]);
    for ids in [
        &[99, 99, 100][..],
        &[98, 99, 100],
        &[98, 99, 99, 8],
        &[98, 99, 99, 100, 99],
        &[1, 2, 3],
    ] {
        assert!(interpret(tokens(ids), &pixels, &"a".repeat(64), &geometry()).is_err());
    }
    assert!(interpret(
        tokens(&[98, 99, 99, 100]),
        &pixels,
        &"b".repeat(64),
        &geometry()
    )
    .is_err());
    for grid in [
        [1, 1i64 << 32, 1i64 << 32],
        [1, -2, 2],
        [2, 2, 2],
        [1, 3, 2],
        [1, 4, 4],
    ] {
        assert!(interpret(
            tokens(&[98, 99, 100]),
            &media(&[grid], vec![0.; 4]),
            &"a".repeat(64),
            &geometry()
        )
        .is_err());
    }
    assert!(interpret(
        tokens(&[98, 99, 99, 100]),
        &media(&[[1, 2, 4]], vec![f32::NAN; 8]),
        &"a".repeat(64),
        &geometry()
    )
    .is_err());
}
#[test]
fn per_image_patches_preserve_order_and_content_identity() {
    let supplied = media(&[[1, 2, 4], [1, 2, 2]], (0..12).map(|n| n as f32).collect());
    let ids = tokens(&[98, 99, 99, 100, 7, 98, 99, 100]);
    let prepared = interpret(ids.clone(), &supplied, &"a".repeat(64), &geometry()).unwrap();
    assert_eq!(
        prepared
            .plan
            .layout()
            .spans()
            .iter()
            .map(|s| (s.start, s.end))
            .collect::<Vec<_>>(),
        vec![(1, 3), (6, 7)]
    );
    for (image, expected) in prepared.images.iter().zip([0..8, 8..12]) {
        let values = image
            .pixels
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, expected.map(|n| n as f32).collect::<Vec<_>>());
    }
    let changed = interpret(
        ids,
        &media(&[[1, 2, 4], [1, 2, 2]], (1..13).map(|n| n as f32).collect()),
        &"a".repeat(64),
        &geometry(),
    )
    .unwrap();
    assert_ne!(prepared.images[0].identity, changed.images[0].identity);
    assert_eq!(
        prepared.images[0].identity,
        prepared.plan.layout().spans()[0].identity
    );
}

#[test]
fn spatial_controls_preserve_merge_order_and_bilinear_plane() {
    use seismic_engine::models::qwen35::preparation::spatial_controls;
    let controls = spatial_controls([1, 4, 6], 2, 5).unwrap();
    assert_eq!(
        &controls.coordinates[..8],
        &[
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3]
        ]
    );
    for (row, &[y, x]) in controls.coordinates.iter().enumerate() {
        let sum: f32 = (0..4).map(|i| controls.coefficients[i][row]).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let actual: f32 = (0..4)
            .map(|i| {
                let index = controls.indices[i][row];
                (2 * (index / 5) + 3 * (index % 5)) as f32 * controls.coefficients[i][row]
            })
            .sum();
        let expected = 2.0 * (y as f32 * 4.0 / 3.0) + 3.0 * (x as f32 * 4.0 / 5.0);
        assert!(
            (actual - expected).abs() < 2e-6,
            "{row}: {actual} vs {expected}"
        );
    }
    let singleton = spatial_controls([1, 1, 1], 1, 1).unwrap();
    assert_eq!(singleton.coordinates, vec![[0, 0]]);
    assert_eq!(singleton.coefficients.map(|v| v[0]), [1., 0., 0., 0.]);
    for (grid, merge, table) in [
        ([2, 4, 6], 2, 5),
        ([1, 3, 6], 2, 5),
        ([1, 4, 6], 0, 5),
        ([1, 4, 6], 2, 0),
        ([1, usize::MAX, 2], 1, 5),
        ([1, 2, 2], 1, usize::MAX),
    ] {
        assert!(spatial_controls(grid, merge, table).is_err());
    }
}
