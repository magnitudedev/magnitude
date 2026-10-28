//! The LFM3-VL projector (`clip.vision.projector_type = lfm3vl`): a tower of
//! gated 3 × 3 depthwise convolution blocks and bidirectional grouped-query
//! attention blocks over each image's patches, embedded by a two-frame linear
//! patchifier and placed by axial rotary alone, then a layer norm and a 2 × 2
//! cell concatenation into a two-stage GELU projector.

use crate::Error;
use magnitude_artifacts::gguf::{Directory, Element, Scalar, Value, PROJECTOR_ARCHITECTURE};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    checked_product, ActivationDType, CellReduction, ImportTransform, MergerStage, RmsNorm,
    RowRange, VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan,
    VisionBlock, VisionDescription, VisionFeedForward, VisionGatedConvolution, VisionLinear,
    VisionMerger, VisionNorm, VisionOperator, VisionPreprocessing, VisionResampling, VisionResize,
    VisionStem, VisionUp, WeightDescriptor,
};

/// Every projector metadata key the family reads, beside `general.*`.
const KEYS: &[&str] = &[
    "clip.has_vision_encoder",
    "clip.use_silu",
    "clip.vision.projector_type",
    "clip.vision.projection_dim",
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.block_count",
    "clip.vision.attention.head_count",
    "clip.vision.attention.head_count_kv",
    "clip.vision.attention.layer_norm_epsilon",
    "clip.vision.spatial_merge_size",
    "clip.vision.image_mean",
    "clip.vision.image_std",
    "clip.vision.image_min_pixels",
    "clip.vision.image_max_pixels",
    "clip.vision.image_resize_algo",
];

/// The tower's axial rotary base (`vision_config.rope_parameters.rope_theta`
/// in the released configuration); the converter does not store it.
const ROTARY_BASE: f64 = 1.0e6;

/// The projector layer norm's epsilon (`projector_layer_norm_eps`); the
/// converter does not store it.
const MERGER_NORM_EPSILON: f64 = 1.0e-6;

/// The frames of one still image the patchifier reads (`temporal_patch_size`):
/// the converter splits its linear into one patch weight per frame.
const PATCH_FRAMES: [&str; 2] = ["v.patch_embd.weight", "v.patch_embd.weight.1"];

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

fn positive(directory: &Directory, key: &str) -> Result<u64, Error> {
    directory
        .value(key)
        .and_then(Value::unsigned)
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{key} must be a positive integer")))
}

fn triple(directory: &Directory, key: &str) -> Result<[f64; 3], Error> {
    let Some(Value::Array(values)) = directory.value(key) else {
        return Err(invalid(format!("{key} must contain three numbers")));
    };
    let values = values
        .iter()
        .map(|value| match value {
            Element::Float(value) if value.is_finite() => Some(value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid(format!("{key} must contain three finite numbers")))?;
    <[f64; 3]>::try_from(values).map_err(|_| invalid(format!("{key} must contain three numbers")))
}

/// Interpret an LFM3-VL projector inventory without importing it.
pub fn describe_projector(directory: &Directory) -> Result<VisionDescription, Error> {
    if directory
        .value("general.architecture")
        .and_then(Value::string)
        != Some(PROJECTOR_ARCHITECTURE)
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory.value("clip.has_vision_encoder") != Some(&Value::Scalar(Scalar::Bool(true)))
        || directory
            .value("clip.vision.projector_type")
            .and_then(Value::string)
            != Some("lfm3vl")
    {
        return Err(invalid("unsupported LFM3-VL projector identity"));
    }
    if let Some(entry) = directory
        .metadata
        .iter()
        .find(|entry| !entry.name.starts_with("general.") && !KEYS.contains(&entry.name.as_str()))
    {
        return Err(invalid(format!(
            "unknown projector metadata {:?}",
            entry.name
        )));
    }
    // The tower's feed-forward is SwiGLU and its pixels are resampled
    // bicubically; anything else is another projector.
    if directory.value("clip.use_silu") != Some(&Value::Scalar(Scalar::Bool(true)))
        || directory
            .value("clip.vision.image_resize_algo")
            .and_then(Value::string)
            != Some("bicubic")
    {
        return Err(invalid("LFM3-VL projector activation or resampling"));
    }
    let depth = positive(directory, "clip.vision.block_count")?;
    let hidden = positive(directory, "clip.vision.embedding_length")?;
    let intermediate = positive(directory, "clip.vision.feed_forward_length")?;
    let heads = positive(directory, "clip.vision.attention.head_count")?;
    let kv_heads = positive(directory, "clip.vision.attention.head_count_kv")?;
    let patch = positive(directory, "clip.vision.patch_size")?;
    let merge = positive(directory, "clip.vision.spatial_merge_size")?;
    let output_hidden = positive(directory, "clip.vision.projection_dim")?;
    let min_pixels = positive(directory, "clip.vision.image_min_pixels")?;
    let max_pixels = positive(directory, "clip.vision.image_max_pixels")?;
    let epsilon = match directory.value("clip.vision.attention.layer_norm_epsilon") {
        Some(Value::Scalar(Scalar::Float(value))) if value.is_finite() && *value > 0.0 => *value,
        _ => return Err(invalid("projector declares no positive layer norm epsilon")),
    };
    let mean = triple(directory, "clip.vision.image_mean")?;
    let std = triple(directory, "clip.vision.image_std")?;
    if hidden % heads != 0 || heads % kv_heads != 0 {
        return Err(invalid("projector heads do not group over the tower width"));
    }
    let width = hidden / heads;
    let product = |dimensions: &[u64]| {
        checked_product(dimensions).map_err(|error| invalid(error.to_string()))
    };

    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let rms = |weight: WeightDescriptor| VisionNorm::Rms {
        weight: Some(weight),
        epsilon,
    };
    let linear = |weight: WeightDescriptor| VisionLinear {
        weight,
        bias: None,
        clamp: None,
    };
    let frames = PATCH_FRAMES
        .into_iter()
        .map(|name| tensor(name, &[hidden, 3, patch, patch]))
        .collect::<Result<Vec<_>, _>>()?;
    let stem_norm = rms(tensor("v.embedding_norm.weight", &[hidden])?);
    let gated_rows = product(&[3, hidden])?;
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        // A block convolves where it stores a convolution and attends
        // otherwise; the tensor inventory is the only source of the pattern.
        let operator = if directory
            .tensor(&format!("{p}conv2d_in_proj.weight"))
            .is_some()
        {
            // `in_proj` stores the B, C and X projections as consecutive row
            // blocks of the tower width.
            let fused = weight("conv2d_in_proj.weight", &[gated_rows, hidden])?;
            let chunk = |block: u64| VisionLinear {
                weight: WeightDescriptor {
                    name: fused.name.clone(),
                    shape: vec![hidden, hidden],
                    transforms: vec![ImportTransform::Rows(RowRange {
                        start: block * hidden,
                        rows: hidden,
                    })],
                },
                bias: None,
                clamp: None,
            };
            let kernel = directory
                .tensor(&format!("{p}conv2d_spatial.weight"))
                .and_then(|tensor| tensor.shape.last().copied())
                .ok_or_else(|| invalid("missing projector convolution kernel"))?;
            VisionOperator::GatedConvolution(VisionGatedConvolution {
                input_gate: chunk(0),
                output_gate: chunk(1),
                value: chunk(2),
                kernel: weight("conv2d_spatial.weight", &[hidden, 1, kernel, kernel])?,
                output: linear(weight("conv2d_out_proj.weight", &[hidden, hidden])?),
            })
        } else {
            let kv_rows = product(&[kv_heads, width])?;
            VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads,
                width,
                query: linear(weight("attn_q.weight", &[hidden, hidden])?),
                key: linear(weight("attn_k.weight", &[kv_rows, hidden])?),
                value: linear(weight("attn_v.weight", &[kv_rows, hidden])?),
                query_norm: Some(RmsNorm {
                    weight: weight("attn_q_norm.weight", &[width])?,
                    epsilon,
                }),
                key_norm: Some(RmsNorm {
                    weight: weight("attn_k_norm.weight", &[width])?,
                    epsilon,
                }),
                value_norm: None,
                rotary_base: Some(ROTARY_BASE),
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: linear(weight("attn_out.weight", &[hidden, hidden])?),
            })
        };
        blocks.push(VisionBlock {
            operator_norm: rms(weight("ln1.weight", &[hidden])?),
            operator,
            operator_post_norm: None,
            feedforward_norm: rms(weight("ln2.weight", &[hidden])?),
            feedforward: VisionFeedForward {
                up: VisionUp::Gated {
                    gate: linear(weight("ffn_gate.weight", &[intermediate, hidden])?),
                    up: linear(weight("ffn_up.weight", &[intermediate, hidden])?),
                },
                activation: VisionActivation::Silu,
                down: linear(weight("ffn_down.weight", &[hidden, intermediate])?),
            },
            feedforward_post_norm: None,
        });
    }
    let merger_norm = VisionNorm::Layer {
        weight: tensor("mm.input_norm.weight", &[hidden])?,
        bias: tensor("mm.input_norm.bias", &[hidden])?,
        epsilon: MERGER_NORM_EPSILON,
    };
    let merger_hidden = product(&[hidden, merge, merge])?;
    let projector_hidden = directory
        .tensor("mm.0.weight")
        .and_then(|tensor| tensor.shape.first().copied())
        .ok_or_else(|| invalid("missing projector merger stage"))?;
    let stages = vec![
        MergerStage {
            linear: VisionLinear {
                weight: tensor("mm.0.weight", &[projector_hidden, merger_hidden])?,
                bias: Some(tensor("mm.0.bias", &[projector_hidden])?),
                clamp: None,
            },
            activation: Some(VisionActivation::GeluErf),
        },
        MergerStage {
            linear: VisionLinear {
                weight: tensor("mm.2.weight", &[output_hidden, projector_hidden])?,
                bias: Some(tensor("mm.2.bias", &[output_hidden])?),
                clamp: None,
            },
            activation: None,
        },
    ];
    tensors.finish()?;
    let description = VisionDescription {
        // f16 intermediates around an F32 residual stream, the precision the
        // vision kernel family measured for its towers.
        activation_dtype: ActivationDType::F16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PixelBounds {
                min_pixels,
                max_pixels,
            },
            resampling: VisionResampling::Bicubic,
            mean,
            std,
            channels: 3,
            patch,
            merge,
        },
        stem: VisionStem::Patch {
            frames,
            bias: None,
            positions: None,
            norm: Some(stem_norm),
            registers: None,
        },
        window: None,
        blocks,
        merger: VisionMerger {
            norm: Some(merger_norm),
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: None,
            stages,
            output_norm: None,
        },
    };
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}
