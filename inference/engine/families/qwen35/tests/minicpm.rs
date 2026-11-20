use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor},
    ArtifactIdentity, ImageProcessor, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_common::headers;
use magnitude_family_contracts::{
    FamilyError, MarkerTokens, ModelDefinition, PositionSampling, TokenPlan, VisionOperator,
    VisionResize,
};
use magnitude_family_qwen35::{inspect_components, minicpm};

/// A released MiniCPM-V 4.7 model with its projector, pinned artifacts outside the catalog.
fn definition(large: bool) -> ModelDefinition {
    let model = if large {
        "minicpm-v-4.7-35b-a3b"
    } else {
        "minicpm-v-4.7-1b"
    };
    inspect_components(
        &headers::artifact(&format!("{model}-q4")),
        Some(&headers::artifact(&format!("{model}-mmproj"))),
        PackageIdentity {
            target: ArtifactIdentity([0; 32]),
            projector: Some(ArtifactIdentity([1; 32])),
        },
    )
    .unwrap()
}

/// A projector's declared feature width.
fn width(projector: &magnitude_artifacts::gguf::Directory) -> u64 {
    projector
        .value("clip.vision.projection_dim")
        .and_then(magnitude_artifacts::gguf::Value::unsigned)
        .unwrap()
}

/// Another model's projector, or one of another width, disagrees with the target, before the
/// MiniCPM-V and Qwen3-VL readers alike; a header that is no projector is unreadable.
#[test]
fn another_models_projector_is_a_mismatch() {
    use magnitude_artifacts::UnsupportedReason;
    let gemma = headers::projector("gemma-4-e4b-it-qat:gguf:q4");
    let own = headers::artifact("minicpm-v-4.7-1b-mmproj");
    for error in [
        minicpm::describe_projector(&gemma, width(&gemma)).unwrap_err(),
        magnitude_family_qwen::describe_projector(&gemma, width(&gemma)).unwrap_err(),
        // Its own kind of projector, at another target's width.
        minicpm::describe_projector(&own, width(&own) + 1).unwrap_err(),
    ] {
        assert!(
            matches!(error.reasons[..], [UnsupportedReason::TargetMismatch { .. }]),
            "{error}"
        );
    }
    let target = headers::artifact("minicpm-v-4.7-1b-q4");
    assert_eq!(
        minicpm::describe_projector(&target, width(&own)).unwrap_err().reasons,
        [UnsupportedReason::Unreadable]
    );
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
                .all(|b| matches!(
                    &b.operator,
                    VisionOperator::Attention(attention)
                        if attention.rotary_base.is_none() && attention.query_norm.is_none()
                ))
        );
        assert_eq!(
            v.stem.positions().expect("position table").sampling,
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
    let mut p = headers::artifact("minicpm-v-4.7-1b-mmproj");
    let original = p.clone();
    p.metadata
        .iter_mut()
        .find(|e| e.name == "clip.vision.projector.scale_factor")
        .unwrap()
        .value = Value::Scalar(Scalar::Unsigned(4));
    assert!(minicpm::describe_projector(&p, width(&p)).is_err());
    p = original;
    let mut t = p.tensors[0].clone();
    t.name = "unexpected.weight".into();
    p.tensors.push(t);
    assert!(minicpm::describe_projector(&p, width(&p)).is_err());
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
        &s.position_blend().expect("position blend").indices[0][..8],
        &[0, 17, 1190, 1207, 35, 52, 1225, 1242]
    );
    let coefficients = &s.position_blend().expect("position blend").coefficients;
    assert!(coefficients[0].iter().all(|c| *c == 1.));
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

/// MiniCPM expands a single image marker into ID tokens, overview span, detail
/// slice spans, and delimiters; `prompt_position_row` maps every source
/// boundary (including post-image message boundaries) directly to its prepared
/// row, and two inputs with the same overview crop but different detail tilings
/// carry distinct overview RoPE coordinates.
#[test]
fn minicpm_source_boundaries_and_overview_coordinates_reflect_detail_tiling() {
    let d = definition(false);
    let adapter = minicpm::inputs::adapter(&d, &Markers).unwrap();
    let prompt = vec![TokenId(10), TokenId(100), TokenId(11), TokenId(12)];
    let plan = |tokens: &[TokenId]| {
        TokenPlan::new(tokens.to_vec(), InputLayout::new(tokens.len(), vec![]).unwrap()).unwrap()
    };

    // Unsliced image: overview only ([0, 0] detail grid).
    let unsliced = adapter
        .prepare(&d, plan(&prompt), &[media(&d, 1, [0, 0])])
        .unwrap();
    // Sliced image: identical overview crop [1, 4, 4] followed by a 2x2 detail
    // grid of [1, 4, 4] slices (canvas [4, 4] instead of [2, 2]).
    let sliced = adapter
        .prepare(&d, plan(&prompt), &[media(&d, 5, [2, 2])])
        .unwrap();

    // Every source boundary maps to the exact prepared row of its token.
    for prepared in [&unsliced, &sliced] {
        for (source, &token) in prompt.iter().enumerate() {
            let row = prepared.prompt_position_row(source).unwrap();
            if token != TokenId(100) {
                assert_eq!(prepared.tokens()[row], token);
            }
        }
        assert_eq!(
            prepared.prompt_position_row(prompt.len()),
            Some(prepared.tokens().len())
        );
        assert_eq!(prepared.prompt_position_row(prompt.len() + 1), None);
    }

    // Both inputs share the same overview crop identity and overview span
    // range, yet their overview RoPE coordinates differ because the sliced
    // input scales overview coordinates across the 2x2 detail canvas.
    let unsliced_overview = &unsliced.layout().spans()[0];
    let sliced_overview = &sliced.layout().spans()[0];
    assert_eq!(unsliced_overview, sliced_overview);
    assert_eq!(
        &unsliced.tokens()[..unsliced_overview.end],
        &sliced.tokens()[..sliced_overview.end]
    );
    assert_ne!(
        &unsliced.coordinates()[unsliced_overview.start..unsliced_overview.end],
        &sliced.coordinates()[sliced_overview.start..sliced_overview.end]
    );
}
