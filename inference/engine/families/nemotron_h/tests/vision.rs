//! Interpretation of the released Nemotron Omni projectors and the image
//! input they take.
//!
//! The headers are the locked catalog components' (`headers::target`,
//! `headers::projector`): the
//! Nemotron 3.5 Bolt 74B-A9B and Super-VL 120B-A12B (UD-Q4_K_XL, every
//! shard) targets and their projectors.

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    media::{DType, PreparedMedia, PreparedTensor},
    ArtifactIdentity, ImageProcessor, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_common::headers::{self, add_tensor, remove_tensor, reshape_tensor, set};
use magnitude_family_contracts::{
    ActivationDType, CellReduction, FamilyError, ImportTransform, MarkerTokens, ModelDefinition,
    ModelFamily, PositionSampling, RowRange, TokenPlan, VisionActivation, VisionAttentionScale,
    VisionAttentionSpan, VisionDescription, VisionNorm, VisionResampling, VisionResize, VisionStem,
    VisionUp,
};
use magnitude_family_nemotron_h::{
    inputs::spatial_controls, inspect_components, projector::describe_projector, NemotronHFamily,
};

const BOLT: &str = "nemotron-3.5-bolt-74b-a9b:gguf:iq2-qat";
const SUPER: &str = "nemotron-3.5-super-vl-120b-a12b:gguf:q4";

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([0; 32]),
        projector: Some(ArtifactIdentity([1; 32])),
    }
}

/// The Super-VL target with its released projector.
fn super_vl(projector: &Directory) -> Result<ModelDefinition, String> {
    let target = headers::target(SUPER);
    inspect_components(&target, Some(projector), identity()).map_err(|error| error.to_string())
}

fn embedding_length(model: &str) -> u64 {
    headers::target(model)
        .value("nemotron_h_moe.embedding_length")
        .and_then(Value::unsigned)
        .unwrap()
}

/// Every role of the C-RADIO tower and the squared-ReLU projection. Bolt
/// stores its query, key and value apart; Super-VL fuses them, stores its
/// position grid with a trailing unit GGML dimension and carries the
/// `vision_final_layernorm` its MTP-trained language model implies.
fn check(vision: &VisionDescription, fused: bool) {
    vision.validate().unwrap();
    assert_eq!(vision.activation_dtype, ActivationDType::BF16);
    assert_eq!(
        (vision.hidden, vision.output_hidden, vision.blocks.len()),
        (1280, 4096, 32)
    );
    let p = &vision.preprocessing;
    assert_eq!(
        p.resize,
        VisionResize::PatchBounds {
            min_patches: 1_024,
            max_patches: 13_312,
        }
    );
    assert_eq!(p.resampling, VisionResampling::Bicubic);
    assert_eq!((p.channels, p.patch, p.merge), (3, 16, 2));
    for (value, expected) in p.mean.iter().zip([0.48145466, 0.4578275, 0.40821073]) {
        assert!((value - expected).abs() < 1e-7);
    }
    for (value, expected) in p.std.iter().zip([0.26862954, 0.26130258, 0.27577711]) {
        assert!((value - expected).abs() < 1e-7);
    }
    assert_eq!(vision.max_cells().unwrap(), 3_328);
    assert_eq!(vision.patch_row_width().unwrap(), 768);
    assert_eq!(vision.window, None);
    let VisionStem::Patch {
        frames,
        bias,
        positions,
        norm,
        registers,
    } = &vision.stem
    else {
        panic!("a patch convolution stem");
    };
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].name, "v.patch_embd.weight");
    assert_eq!(frames[0].shape, [1280, 3, 16, 16]);
    assert!(bias.is_none() && norm.is_none());
    assert_eq!(
        positions.sampling,
        PositionSampling::CroppedSquare { side: 128 }
    );
    assert_eq!(positions.table.shape, [16_384, 1280]);
    assert_eq!(
        positions.table.transforms,
        if fused {
            vec![ImportTransform::DropUnitAxis]
        } else {
            Vec::new()
        }
    );
    let registers = registers.as_ref().unwrap();
    assert_eq!(
        (registers.name.as_str(), registers.shape.as_slice()),
        ("v.class_embd", [10u64, 1280].as_slice())
    );
    assert_eq!(vision.stem.registers(), 10);
    for (index, block) in vision.blocks.iter().enumerate() {
        let attention = &block.attention;
        assert_eq!((attention.heads, attention.width), (16, 80));
        assert_eq!(attention.rotary_base, None);
        assert_eq!(attention.scale, VisionAttentionScale::InverseSqrtWidth);
        assert_eq!(attention.span, VisionAttentionSpan::Full);
        assert!(attention.query_norm.is_none() && attention.value_norm.is_none());
        for (part, linear) in [&attention.query, &attention.key, &attention.value]
            .into_iter()
            .enumerate()
        {
            assert_eq!(linear.weight.shape, [1280, 1280]);
            let bias = linear.bias.as_ref().unwrap();
            assert_eq!(bias.shape, [1280]);
            if fused {
                let rows = ImportTransform::Rows(RowRange {
                    start: part as u64 * 1280,
                    rows: 1280,
                });
                assert_eq!(linear.weight.name, format!("v.blk.{index}.attn_qkv.weight"));
                assert_eq!(linear.weight.transforms, [rows.clone()]);
                assert_eq!(bias.transforms, [rows]);
            } else {
                let name = ["q", "k", "v"][part];
                assert_eq!(
                    linear.weight.name,
                    format!("v.blk.{index}.attn_{name}.weight")
                );
                assert!(linear.weight.transforms.is_empty());
            }
        }
        assert!(attention.output.bias.is_some());
        for norm in [&block.attention_norm, &block.feedforward_norm] {
            assert!(
                matches!(norm, VisionNorm::Layer { epsilon, .. } if (epsilon - 1e-6).abs() < 1e-12)
            );
        }
        assert!(block.attention_post_norm.is_none() && block.feedforward_post_norm.is_none());
        assert_eq!(block.feedforward.activation, VisionActivation::GeluErf);
        let VisionUp::Plain(up) = &block.feedforward.up else {
            panic!("a plain feed-forward");
        };
        assert_eq!(up.weight.shape, [5120, 1280]);
        assert_eq!(block.feedforward.down.weight.shape, [1280, 5120]);
    }
    let merger = &vision.merger;
    assert_eq!(merger.norm.is_some(), fused);
    if let Some(VisionNorm::Layer {
        weight, epsilon, ..
    }) = &merger.norm
    {
        assert_eq!(weight.name, "v.post_ln.weight");
        assert!((epsilon - 1e-6).abs() < 1e-12);
    }
    assert_eq!(merger.reduction, CellReduction::Concatenate);
    assert!(merger.standardize.is_none() && merger.output_norm.is_none());
    let Some(VisionNorm::Rms {
        weight: Some(weight),
        epsilon,
    }) = &merger.projection_norm
    else {
        panic!("a weighted RMS norm of each cell");
    };
    assert_eq!(
        (weight.name.as_str(), weight.shape.as_slice(), *epsilon),
        ("mm.model.mlp.0.weight", [5120u64].as_slice(), 1e-5)
    );
    let [up, down] = merger.stages.as_slice() else {
        panic!("two projection stages");
    };
    assert_eq!(up.linear.weight.shape, [20_480, 5120]);
    assert_eq!(up.activation, Some(VisionActivation::ReluSquared));
    assert_eq!(down.linear.weight.shape, [4096, 20_480]);
    assert_eq!(down.activation, None);
    assert!(up.linear.bias.is_none() && down.linear.bias.is_none());
}

#[test]
fn both_released_projectors_bind_the_radio_tower_and_projection() {
    for (model, fused) in [(BOLT, false), (SUPER, true)] {
        let directory = headers::projector(model);
        let vision = describe_projector(&directory).unwrap();
        check(&vision, fused);
        // Features have the width of the target each projector ships with.
        assert_eq!(vision.output_hidden, embedding_length(model));
        // Every stored tensor but the video kernel has a role.
        let named = vision
            .weights()
            .into_iter()
            .map(|(_, weight)| weight.name.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(named.len() + 1, directory.tensors.len());
        assert!(!named.contains("v.patch_embd.weight.1"));
    }
}

#[test]
fn both_targets_inspect_with_their_projectors_and_place_images() {
    for (model, fused) in [(BOLT, false), (SUPER, true)] {
        let definition = inspect_components(
            &headers::target(model),
            Some(&headers::projector(model)),
            identity(),
        )
        .unwrap();
        definition.validate().unwrap();
        check(definition.vision.as_ref().unwrap(), fused);
        assert_eq!(
            NemotronHFamily.media_placeholder(&definition),
            Some("<img><image></img>")
        );
    }
}

#[test]
fn unknown_misshapen_or_mismatched_projectors_are_refused() {
    let original = headers::projector(BOLT);
    let refused = |edit: &dyn Fn(&mut Directory)| {
        let mut directory = original.clone();
        edit(&mut directory);
        describe_projector(&directory).unwrap_err()
    };
    refused(&|d| {
        set(
            d,
            "clip.projector_type",
            Value::Scalar(Scalar::String("nemotron_v2_vl".into())),
        )
    });
    refused(&|d| {
        set(
            d,
            "clip.vision.unexpected",
            Value::Scalar(Scalar::Unsigned(1)),
        )
    });
    refused(&|d| set(d, "clip.use_gelu", Value::Scalar(Scalar::Bool(false))));
    refused(&|d| {
        set(
            d,
            "clip.vision.image_min_pixels",
            Value::Scalar(Scalar::Unsigned(1000)),
        )
    });
    refused(&|d| add_tensor(d, "v.unexpected.weight", &[4]));
    refused(&|d| remove_tensor(d, "v.class_embd"));
    refused(&|d| reshape_tensor(d, "v.position_embd.weight", &[16_383, 1280]));
    refused(&|d| reshape_tensor(d, "mm.model.mlp.0.weight", &[4096]));

    // A projector whose features are narrower than the target's width.
    let mut narrow = headers::projector(SUPER);
    set(
        &mut narrow,
        "clip.vision.projection_dim",
        Value::Scalar(Scalar::Unsigned(2048)),
    );
    reshape_tensor(&mut narrow, "mm.model.mlp.3.weight", &[2048, 20_480]);
    describe_projector(&narrow).unwrap();
    let error = super_vl(&narrow).unwrap_err();
    assert!(error.contains("decoder's width"), "{error}");
}

/// The C-RADIO cropped position embedding: PyTorch's bilinear
/// `interpolate` of the 128 × 128 grid to a square of the longer side, then
/// the grid's top-left rows and columns. The table row index stands for the
/// table value, so each blend is the sampled coordinate; the expected values
/// are PyTorch 2.10's (`F.interpolate(t, (m, m), mode="bilinear",
/// align_corners=False)[..., :h, :w]` of `t = arange(128²)`).
#[test]
fn positions_sample_the_cropped_square_grid() {
    let definition = super_vl(&headers::projector(SUPER)).unwrap();
    let expected_4_by_6 = [
        1311.5001, 1332.8334, 1354.1669, 1375.5001, 1396.8335, 1418.1667, 4042.1667, 4063.5,
        4084.8335, 4106.1665, 4127.5, 4148.8335, 6772.834, 6794.167, 6815.5005, 6836.8335,
        6858.167, 6879.5005, 9503.5, 9524.834, 9546.167, 9567.501, 9588.834, 9610.167,
    ];
    let expected_6_by_4 = [
        1311.5001, 1332.8334, 1354.1669, 1375.5001, 4042.1667, 4063.5, 4084.8335, 4106.1665,
        6772.834, 6794.167, 6815.5005, 6836.8335, 9503.5, 9524.834, 9546.167, 9567.501, 12234.167,
        12255.5, 12276.834, 12298.167, 14964.834, 14986.167, 15007.501, 15028.833,
    ];
    for ([h, w], expected) in [([4, 6], expected_4_by_6), ([6, 4], expected_6_by_4)] {
        let controls = spatial_controls(&definition, [1, h, w]).unwrap();
        assert_eq!(controls.rows(), h * w);
        assert!(controls
            .attention_coordinates()
            .iter()
            .all(|c| *c == [0, 0]));
        assert!(controls.window_ranges().is_empty());
        let (indices, coefficients) = (
            controls.interpolation_indices(),
            controls.interpolation_coefficients(),
        );
        // Rows arrive 2 × 2 cell by cell.
        let mut row = 0;
        for cell_row in 0..h / 2 {
            for cell_col in 0..w / 2 {
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let (y, x) = (2 * cell_row + dy, 2 * cell_col + dx);
                    let blended = (0..4)
                        .map(|plane| indices[plane][row] as f64 * coefficients[plane][row] as f64)
                        .sum::<f64>();
                    let weights = (0..4).map(|plane| coefficients[plane][row]).sum::<f32>();
                    assert!((weights - 1.0).abs() < 1e-6);
                    let want = f64::from(expected[y * w + x]);
                    assert!(
                        (blended - want).abs() < 2e-2,
                        "{h}x{w} ({y}, {x}): {blended} != {want}"
                    );
                    row += 1;
                }
            }
        }
    }
    // A grid the table's size reads its rows exactly.
    let controls = spatial_controls(&definition, [1, 128, 128]).unwrap();
    assert!(controls.interpolation_coefficients()[0]
        .iter()
        .all(|c| *c == 1.0));
    assert_eq!(&controls.interpolation_indices()[0][..4], &[0, 1, 128, 129]);
    assert!(spatial_controls(&definition, [1, 3, 4]).is_err());
}

struct Markers;

impl MarkerTokens for Markers {
    fn marker(&self, text: &str) -> Result<TokenId, FamilyError> {
        match text {
            "<image>" => Ok(TokenId(18)),
            "<img>" => Ok(TokenId(19)),
            "</img>" => Ok(TokenId(20)),
            _ => Err(FamilyError::invalid(format!("no marker {text:?}"))),
        }
    }
}

fn media(definition: &ModelDefinition, grids: &[[i64; 3]]) -> PreparedMedia {
    let vision = definition.vision.as_ref().unwrap();
    let processor = ImageProcessor::new(vision.image_processor_config().unwrap()).unwrap();
    let width = vision.patch_row_width().unwrap() as usize;
    let rows = grids
        .iter()
        .map(|[t, h, w]| (t * h * w) as usize)
        .sum::<usize>();
    let pixels = (0..rows * width)
        .flat_map(|index| ((index % 251) as f32 / 251.0).to_le_bytes())
        .collect();
    PreparedMedia::new(
        processor.identity().into(),
        vec![
            PreparedTensor::new("pixel_values".into(), DType::F32, vec![rows, width], pixels)
                .unwrap(),
            PreparedTensor::new(
                "image_grid_thw".into(),
                DType::I64,
                vec![grids.len(), 3],
                grids
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

/// Each placeholder expands to one `<image>` row per 2 × 2 cell between
/// `<img>` and `</img>`; every row, text or media, takes its absolute
/// position.
#[test]
fn images_expand_between_their_markers_at_absolute_positions() {
    let definition = super_vl(&headers::projector(SUPER)).unwrap();
    let adapter = NemotronHFamily
        .input_adapter(&definition, &Markers)
        .unwrap();
    let tokens = [1, 19, 18, 20, 7, 19, 18, 20, 2].map(TokenId).to_vec();
    let plan = TokenPlan::new(
        tokens.clone(),
        InputLayout::new(tokens.len(), Vec::new()).unwrap(),
    )
    .unwrap();
    let prepared = adapter
        .prepare(
            &definition,
            plan,
            &[media(&definition, &[[1, 4, 6], [1, 2, 2]])],
        )
        .unwrap();
    let expected = [1, 19]
        .into_iter()
        .chain([18; 6])
        .chain([20, 7, 19, 18, 20, 2])
        .map(TokenId)
        .collect::<Vec<_>>();
    assert_eq!(prepared.tokens(), expected);
    let spans = prepared.layout().spans();
    assert_eq!(
        spans
            .iter()
            .map(|span| (span.start, span.end))
            .collect::<Vec<_>>(),
        [(2, 8), (11, 12)]
    );
    assert_eq!(prepared.vision().len(), 2);
    assert_eq!(prepared.continuation(), expected.len());
    for (position, coordinates) in prepared.coordinates().iter().enumerate() {
        assert_eq!(*coordinates, [position as i32; 3]);
    }
    // A placeholder without its image, or an image without its placeholder,
    // misaligns the input.
    let one = TokenPlan::new(
        tokens[..4].to_vec(),
        InputLayout::new(4, Vec::new()).unwrap(),
    )
    .unwrap();
    assert!(adapter
        .prepare(
            &definition,
            one,
            &[media(&definition, &[[1, 4, 6], [1, 2, 2]])]
        )
        .is_err());
}

/// A target without a projector keeps its text-only input.
#[test]
fn a_text_only_definition_takes_no_markers_or_images() {
    let target = headers::target(SUPER);
    let definition = inspect_components(
        &target,
        None,
        PackageIdentity {
            target: ArtifactIdentity([0; 32]),
            projector: None,
        },
    )
    .unwrap();
    assert!(definition.vision.is_none());
    assert_eq!(NemotronHFamily.media_placeholder(&definition), None);
    struct NoMarkers;
    impl MarkerTokens for NoMarkers {
        fn marker(&self, text: &str) -> Result<TokenId, FamilyError> {
            Err(FamilyError::invalid(format!("no marker {text:?}")))
        }
    }
    NemotronHFamily
        .input_adapter(&definition, &NoMarkers)
        .unwrap();
}
