use magnitude_artifacts::TokenId;
use magnitude_model_qwen35::inputs::{spatial_controls, QwenImageTokens};

#[test]
fn image_markers_require_distinct_addressable_tokens() {
    assert!(QwenImageTokens::new(TokenId(99), TokenId(98), TokenId(100)).is_ok());
    assert!(QwenImageTokens::new(TokenId(99), TokenId(99), TokenId(100)).is_err());
    assert!(QwenImageTokens::new(TokenId(u32::MAX), TokenId(98), TokenId(100)).is_err());
}

#[test]
fn spatial_controls_follow_merge_group_order_and_normalized_bilinear_weights() {
    let controls = spatial_controls([1, 4, 4], 2, 4).unwrap();
    assert_eq!(controls.attention_coordinates().len(), 16);
    assert_eq!(controls.patch_order(), &(0..16).collect::<Vec<_>>());
    assert_eq!(
        &controls.attention_coordinates()[..8],
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
    for row in 0..16 {
        let sum = controls
            .interpolation_coefficients()
            .iter()
            .map(|plane| plane[row])
            .sum::<f32>();
        assert!((sum - 1.0).abs() <= f32::EPSILON * 4.0);
        assert!(controls
            .interpolation_indices()
            .iter()
            .all(|plane| (0..16).contains(&plane[row])));
    }
}
