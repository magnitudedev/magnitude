use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor},
    ArtifactIdentity, ImageProcessor, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_common::headers;
use magnitude_family_contracts::{
    FamilyError, MarkerTokens, ModelDefinition, PositionSampling, TokenPlan, VisionResize,
};
use magnitude_family_qwen35::{inspect_components, minicpm};

fn definition(large: bool) -> ModelDefinition {
    let model = if large {
        "minicpm-v-4.7-35b-a3b"
    } else {
        "minicpm-v-4.7-1b"
    };
    inspect_components(
        &headers::target(&format!("{model}:gguf:q4")),
        Some(&headers::projector(&format!("{model}:gguf:q4"))),
        PackageIdentity {
            target: ArtifactIdentity([0; 32]),
            projector: Some(ArtifactIdentity([1; 32])),
        },
    )
    .unwrap()
}

#[test]
fn both_released_projectors_bind_the_exact_tower_and_merger() {
    for large in [false, true] {
        let d = definition(large);
        d.validate().unwrap();
        let v = d.vision.as_ref().unwrap();
        assert_eq!(v.hidden, 1152);
        assert_eq!(v.blocks.len(), 27);
        assert_eq!(v.output_hidden, if large { 2048 } else { 1024 });
        assert!(
            v.blocks
                .iter()
                .all(|b| b.attention.rotary_base.is_none() && b.attention.query_norm.is_none())
        );
        assert_eq!(
            v.stem.positions().sampling,
            PositionSampling::Buckets { side: 70 }
        );
        assert!(v.merger.norm.is_some() && v.merger.projection_norm.is_some());
        assert_eq!(v.preprocessing.merge, 2);
        assert_eq!(
            v.preprocessing.resize,
            VisionResize::Slices {
                resolution: 448,
                max_slices: 9
            }
        );
    }
}

#[test]
fn unsupported_compression_and_unknown_weights_are_refused() {
    use magnitude_artifacts::gguf::{Scalar, Value};
    let mut p = headers::projector("minicpm-v-4.7-1b:gguf:q4");
    let original = p.clone();
    p.metadata
        .iter_mut()
        .find(|e| e.name == "clip.vision.projector.scale_factor")
        .unwrap()
        .value = Value::Scalar(Scalar::Unsigned(4));
    assert!(minicpm::describe_projector(&p).is_err());
    p = original;
    let mut t = p.tensors[0].clone();
    t.name = "unexpected.weight".into();
    p.tensors.push(t);
    assert!(minicpm::describe_projector(&p).is_err());
}

struct Markers;
impl MarkerTokens for Markers {
    fn marker(&self, text: &str) -> Result<TokenId, FamilyError> {
        let id = match text {
            "<|image_pad|>" => 100,
            "<image>" => 101,
            "</image>" => 102,
            "<slice>" => 103,
            "</slice>" => 104,
            "<image_id>" => 105,
            "</image_id>" => 106,
            "\n" => 198,
            _ if text.len() == 1 => {
                200 + text
                    .parse::<u32>()
                    .map_err(|_| FamilyError::invalid("unknown marker"))?
            }
            _ => return Err(FamilyError::invalid("non-atomic marker")),
        };
        Ok(TokenId(id))
    }
}
fn media(d: &ModelDefinition, crops: usize, layout: [i64; 2]) -> PreparedMedia {
    let p =
        ImageProcessor::new(d.vision.as_ref().unwrap().image_processor_config().unwrap()).unwrap();
    let width = d.vision.as_ref().unwrap().patch_row_width().unwrap() as usize;
    PreparedMedia::new(
        p.identity().into(),
        vec![
            PreparedTensor::new(
                "pixel_values".into(),
                DType::F32,
                vec![crops * 16, width],
                vec![0; crops * 16 * width * 4],
            )
            .unwrap(),
            PreparedTensor::new(
                "image_grid_thw".into(),
                DType::I64,
                vec![crops, 3],
                [1i64, 4, 4]
                    .repeat(crops)
                    .into_iter()
                    .flat_map(i64::to_le_bytes)
                    .collect(),
            )
            .unwrap(),
            PreparedTensor::new(
                "image_slice_grid".into(),
                DType::I64,
                vec![1, 2],
                layout.into_iter().flat_map(i64::to_le_bytes).collect(),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}
fn plan() -> TokenPlan {
    TokenPlan::new(
        vec![TokenId(1), TokenId(100), TokenId(2)],
        InputLayout::new(3, Vec::new()).unwrap(),
    )
    .unwrap()
}

#[test]
fn crop_positions_share_the_canvas_and_repeated_crops_share_encoding() {
    let d = definition(false);
    let adapter = minicpm::inputs::adapter(&d, &Markers).unwrap();
    let p = adapter
        .prepare(&d, plan(), &[media(&d, 5, [2, 2])])
        .unwrap();
    assert_eq!(p.layout().spans().len(), 5);
    assert_eq!(p.vision().len(), 1);
    let spans = p.layout().spans();
    let positions = p.coordinates();
    assert_eq!(
        &positions[spans[0].start..spans[0].end],
        &[[4, 4, 4], [4, 4, 7], [4, 7, 4], [4, 7, 7]]
    );
    assert_eq!(
        &positions[spans[1].start..spans[1].end],
        &[[4, 4, 4], [4, 4, 5], [4, 5, 4], [4, 5, 5]]
    );
    assert_eq!(
        &positions[spans[4].start..spans[4].end],
        &[[4, 6, 6], [4, 6, 7], [4, 7, 6], [4, 7, 7]]
    );
    assert_eq!(positions.last(), Some(&[9, 9, 9]));
    assert_eq!(p.continuation(), 10);
    assert!(
        adapter
            .prepare(&d, plan(), &[media(&d, 3, [2, 2])])
            .is_err()
    );
}

#[test]
fn learned_bucket_indices_and_cell_order_are_exact() {
    let s = minicpm::inputs::spatial_controls([1, 4, 4], 70).unwrap();
    assert_eq!(
        &s.interpolation_indices()[0][..8],
        &[0, 17, 1190, 1207, 35, 52, 1225, 1242]
    );
    assert!(s.interpolation_coefficients()[0].iter().all(|c| *c == 1.));
    assert!(s.attention_coordinates().iter().all(|c| *c == [0, 0]));
    assert!(minicpm::inputs::spatial_controls([1, 3, 4], 70).is_err());
}

#[test]
fn sixteen_images_use_atomic_digits_for_multi_token_ids() {
    let d = definition(false);
    let original = media(&d, 16, [0, 0]);
    let mut tensors = original.tensors()[..2].to_vec();
    tensors.push(
        PreparedTensor::new(
            "image_slice_grid".into(),
            DType::I64,
            vec![16, 2],
            vec![0; 16 * 2 * 8],
        )
        .unwrap(),
    );
    let images = PreparedMedia::new(original.processor().into(), tensors).unwrap();
    let tokens = TokenPlan::new(
        vec![TokenId(100); 16],
        InputLayout::new(16, Vec::new()).unwrap(),
    )
    .unwrap();
    let prepared = minicpm::inputs::adapter(&d, &Markers)
        .unwrap()
        .prepare(&d, tokens, &[images])
        .unwrap();
    assert_eq!(prepared.layout().spans().len(), 16);
    assert_eq!(prepared.vision().len(), 1);
    assert!(
        prepared
            .tokens()
            .windows(4)
            .any(|tokens| tokens == [TokenId(105), TokenId(201), TokenId(200), TokenId(106)])
    );
}
