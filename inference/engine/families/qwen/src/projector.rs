//! The Qwen3-VL projector (`qwen3vl_merger`).

use crate::{Error, product};
use magnitude_artifacts::gguf::{Directory, Element, PROJECTOR_ARCHITECTURE, Scalar, Value};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    ActivationDType, CellReduction, ImportTransform, MergerStage, PositionSampling, RowRange,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm, VisionOperator,
    VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem, VisionUp,
    WeightDescriptor,
};

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

fn number(directory: &Directory, key: &str) -> Result<f64, Error> {
    let value = match directory.value(key) {
        Some(Value::Scalar(Scalar::Float(value))) => *value,
        Some(Value::Scalar(Scalar::Unsigned(value))) => *value as f64,
        Some(Value::Scalar(Scalar::Signed(value))) => *value as f64,
        _ => return Err(invalid(format!("{key} must be numeric"))),
    };
    if !value.is_finite() {
        return Err(invalid(format!("{key} must be finite")));
    }
    Ok(value)
}

fn triple(directory: &Directory, key: &str) -> Result<[f64; 3], Error> {
    let Some(Value::Array(values)) = directory.value(key) else {
        return Err(invalid(format!("{key} must contain three numbers")));
    };
    let values = values
        .iter()
        .map(|value| match value {
            Element::Float(value) if value.is_finite() => Some(value),
            Element::Unsigned(value) => Some(value as f64),
            Element::Signed(value) => Some(value as f64),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid(format!("{key} must contain three finite numbers")))?;
    <[f64; 3]>::try_from(values).map_err(|_| invalid(format!("{key} must contain three numbers")))
}

/// Interpret a validated Qwen3-VL projector inventory without importing it.
pub fn describe_projector(directory: &Directory) -> Result<VisionDescription, Error> {
    if directory
        .value("general.architecture")
        .and_then(Value::string)
        != Some(PROJECTOR_ARCHITECTURE)
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory
            .value("clip.has_vision_encoder")
            .and_then(|value| match value {
                Value::Scalar(Scalar::Bool(value)) => Some(*value),
                _ => None,
            })
            != Some(true)
        || directory
            .value("clip.projector_type")
            .and_then(Value::string)
            != Some("qwen3vl_merger")
    {
        return Err(invalid("unsupported Qwen3-VL projector identity"));
    }
    let depth = positive(directory, "clip.vision.block_count")?;
    let hidden = positive(directory, "clip.vision.embedding_length")?;
    let intermediate = positive(directory, "clip.vision.feed_forward_length")?;
    let heads = positive(directory, "clip.vision.attention.head_count")?;
    let patch = positive(directory, "clip.vision.patch_size")?;
    let merge = positive(directory, "clip.vision.spatial_merge_size")?;
    let image_size = positive(directory, "clip.vision.image_size")?;
    let output_hidden = positive(directory, "clip.vision.projection_dim")?;
    let epsilon = number(directory, "clip.vision.attention.layer_norm_epsilon")?;
    let mean = triple(directory, "clip.vision.image_mean")?;
    let std = triple(directory, "clip.vision.image_std")?;
    let pixel_bound = |key: &str, default: u64| -> Result<u64, Error> {
        match directory.value(key) {
            None => Ok(default),
            Some(value) => value
                .unsigned()
                .filter(|bound| *bound > 0)
                .ok_or_else(|| invalid(format!("{key} must be a positive integer"))),
        }
    };
    let min_pixels = pixel_bound("clip.vision.image_min_pixels", 65_536)?;
    let max_pixels = pixel_bound("clip.vision.image_max_pixels", 16_777_216)?;
    if let Some(value) = directory.value("clip.use_gelu") {
        if !matches!(value, Value::Scalar(Scalar::Bool(_))) {
            return Err(invalid("clip.use_gelu must be boolean"));
        }
    }
    if let Some(deepstack) = directory.value("clip.vision.is_deepstack_layers") {
        if !matches!(deepstack, Value::Array(values) if values.len() as u64 == depth && values.iter().all(|value| matches!(value, Element::Bool(false))))
        {
            return Err(invalid("deepstack projector layers are unsupported"));
        }
    }

    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let patch_frames = ["v.patch_embd.weight", "v.patch_embd.weight.1"]
        .into_iter()
        .map(|name| tensor(name, &[hidden, 3, patch, patch]))
        .collect::<Result<Vec<_>, _>>()?;
    let patch_bias = tensor("v.patch_embd.bias", &[hidden])?;
    let position = directory
        .tensor("v.position_embd.weight")
        .ok_or_else(|| invalid("missing projector position embedding"))?;
    if position.shape.len() != 2 || position.shape[1] != hidden {
        return Err(invalid("invalid projector position embedding shape"));
    }
    let table_rows = position.shape[0];
    let table_side = (table_rows as f64).sqrt() as u64;
    if table_side.checked_mul(table_side) != Some(table_rows) {
        return Err(invalid("projector position table is not square"));
    }
    if image_size.checked_rem(patch) != Some(0) || image_size / patch != table_side {
        return Err(invalid(
            "projector position table does not cover the nominal image",
        ));
    }
    let table = tensor("v.position_embd.weight", &[table_rows, hidden])?;
    if heads == 0 || hidden % heads != 0 || (hidden / heads) % 4 != 0 {
        return Err(invalid(
            "projector heads do not split into 2-D rotary heads",
        ));
    }
    let layer_norm = |weight: WeightDescriptor, bias: WeightDescriptor| VisionNorm::Layer {
        weight,
        bias,
        epsilon,
    };
    // One stored QKV projection, imported as three row ranges.
    let rows = |weight: &WeightDescriptor, part: u64| WeightDescriptor {
        name: weight.name.clone(),
        shape: {
            let mut shape = weight.shape.clone();
            shape[0] = hidden;
            shape
        },
        transforms: vec![ImportTransform::Rows(RowRange {
            start: part * hidden,
            rows: hidden,
        })],
    };
    let qkv_rows = product(&[3, hidden])?;
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let qkv = weight("attn_qkv.weight", &[qkv_rows, hidden])?;
        let qkv_bias = weight("attn_qkv.bias", &[qkv_rows])?;
        let split = |part| VisionLinear {
            weight: rows(&qkv, part),
            bias: Some(rows(&qkv_bias, part)),
            clamp: None,
        };
        blocks.push(VisionBlock {
            operator_norm: layer_norm(
                weight("ln1.weight", &[hidden])?,
                weight("ln1.bias", &[hidden])?,
            ),
            operator: VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads: heads,
                width: hidden / heads,
                query: split(0),
                key: split(1),
                value: split(2),
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: Some(10_000.0),
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: VisionLinear {
                    weight: weight("attn_out.weight", &[hidden, hidden])?,
                    bias: Some(weight("attn_out.bias", &[hidden])?),
                    clamp: None,
                },
            }),
            operator_post_norm: None,
            feedforward_norm: layer_norm(
                weight("ln2.weight", &[hidden])?,
                weight("ln2.bias", &[hidden])?,
            ),
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(VisionLinear {
                    weight: weight("ffn_up.weight", &[intermediate, hidden])?,
                    bias: Some(weight("ffn_up.bias", &[intermediate])?),
                    clamp: None,
                }),
                activation: VisionActivation::GeluTanh,
                down: VisionLinear {
                    weight: weight("ffn_down.weight", &[hidden, intermediate])?,
                    bias: Some(weight("ffn_down.bias", &[hidden])?),
                    clamp: None,
                },
            },
            feedforward_post_norm: None,
        });
    }
    let merger_norm = layer_norm(
        tensor("v.post_ln.weight", &[hidden])?,
        tensor("v.post_ln.bias", &[hidden])?,
    );
    let merger_hidden = product(&[hidden, merge, merge])?;
    let stages = vec![
        MergerStage {
            linear: VisionLinear {
                weight: tensor("mm.0.weight", &[merger_hidden, merger_hidden])?,
                bias: Some(tensor("mm.0.bias", &[merger_hidden])?),
                clamp: None,
            },
            activation: Some(VisionActivation::GeluErf),
        },
        MergerStage {
            linear: VisionLinear {
                weight: tensor("mm.2.weight", &[output_hidden, merger_hidden])?,
                bias: Some(tensor("mm.2.bias", &[output_hidden])?),
                clamp: None,
            },
            activation: None,
        },
    ];
    tensors.finish()?;
    let description = VisionDescription {
        // f16 intermediates around an F32 residual stream: 1.9% relative
        // RMS from an F64 forward of the 4B projector, where bf16 puts the
        // features 12.7% away (see the vision kernel family).
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
            frames: patch_frames,
            bias: Some(patch_bias),
            positions: Some(VisionPositions {
                table,
                sampling: PositionSampling::AlignedCorners { side: table_side },
            }),
            norm: None,
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
