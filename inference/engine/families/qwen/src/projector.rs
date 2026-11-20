//! The Qwen3-VL projector (`qwen3vl_merger`).

use crate::{Error, product};
use magnitude_artifacts::gguf::Directory;
use magnitude_family_common::{ProjectorMetadata, projector_type, Tensors};
use magnitude_family_contracts::{
    ActivationDType, CellReduction, MergerStage, PositionSampling,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm, VisionOperator,
    VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem, VisionUp,
    WeightDescriptor,
};

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

/// Interpret a validated Qwen3-VL projector inventory for a target of
/// `target_width` without importing it.
pub fn describe_projector(
    directory: &Directory,
    target_width: u64,
) -> Result<Box<VisionDescription>, Error> {
    projector_type(directory, &[("qwen3vl_merger", ())], target_width)?;
    let meta = ProjectorMetadata::open(directory);
    let depth = meta.bounded_block_count("clip.vision.block_count")?;
    let hidden = meta.integer("clip.vision.embedding_length")?;
    let intermediate = meta.integer("clip.vision.feed_forward_length")?;
    let heads = meta.integer("clip.vision.attention.head_count")?;
    let patch = meta.integer("clip.vision.patch_size")?;
    let merge = meta.integer("clip.vision.spatial_merge_size")?;
    let image_size = meta.optional_integer("clip.vision.image_size")?;
    let output_hidden = meta.integer("clip.vision.projection_dim")?;
    let epsilon = meta.positive_number("clip.vision.attention.layer_norm_epsilon")?;
    let mean = meta.f64_array("clip.vision.image_mean")?;
    let std = meta.f64_array("clip.vision.image_std")?;
    let min_pixels = meta
        .optional_integer("clip.vision.image_min_pixels")?
        .unwrap_or(65_536);
    let max_pixels = meta
        .optional_integer("clip.vision.image_max_pixels")?
        .unwrap_or(16_777_216);
    meta.invariant_flag("clip.use_gelu", true, "true (GELU feed-forward)")?;
    if meta
        .optional_bool_array("clip.vision.is_deepstack_layers", depth)?
        .is_some_and(|layers| layers.into_iter().any(|enabled| enabled))
    {
        return Err(invalid("deepstack projector layers are unsupported"));
    }
    meta.finish()?;

    let mut tensors = Tensors::new(directory);
    let (table_rows, table_cols) = tensors.matrix_or_batched_shape("v.position_embd.weight")?;
    if table_cols != hidden {
        return Err(invalid("invalid projector position embedding shape"));
    }
    let table_side = (table_rows as f64).sqrt() as u64;
    if table_side.checked_mul(table_side) != Some(table_rows) {
        return Err(invalid("projector position table is not square"));
    }
    if let Some(image_size) = image_size {
        if image_size.checked_rem(patch) != Some(0) || image_size / patch != table_side {
            return Err(invalid(
                "projector position table does not cover the nominal image",
            ));
        }
    }
    let table = tensors.bind_matrix_or_batched("v.position_embd.weight", table_rows, hidden)?;
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let patch_frames = ["v.patch_embd.weight", "v.patch_embd.weight.1"]
        .into_iter()
        .map(|name| tensor(name, &[hidden, 3, patch, patch]))
        .collect::<Result<Vec<_>, _>>()?;
    let patch_bias = tensor("v.patch_embd.bias", &[hidden])?;
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
    // One stored QKV projection, imported as three row ranges (preserving any companion scale).
    let rows = |weight: &WeightDescriptor, part: u64| {
        weight
            .slice_rows(part * hidden, hidden)
            .map_err(|error| Error::invalid(error.to_string()))
    };
    let qkv_rows = product(&[3, hidden])?;
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let qkv = weight("attn_qkv.weight", &[qkv_rows, hidden])?;
        let qkv_bias = weight("attn_qkv.bias", &[qkv_rows])?;
        let split = |part| -> Result<VisionLinear, Error> {
            Ok(VisionLinear {
                weight: rows(&qkv, part)?,
                bias: Some(rows(&qkv_bias, part)?),
                clamp: None,
            })
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
                query: split(0)?,
                key: split(1)?,
                value: split(2)?,
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
            norm: None,
            linear: VisionLinear {
                weight: tensor("mm.0.weight", &[merger_hidden, merger_hidden])?,
                bias: Some(tensor("mm.0.bias", &[merger_hidden])?),
                clamp: None,
            },
            activation: Some(VisionActivation::GeluErf),
        },
        MergerStage {
            norm: None,
            linear: VisionLinear {
                weight: tensor("mm.2.weight", &[output_hidden, merger_hidden])?,
                bias: Some(tensor("mm.2.bias", &[output_hidden])?),
                clamp: None,
            },
            activation: None,
        },
    ];
    tensors.finish()?;
    let description = Box::new(VisionDescription {
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
            feed_forward: None,
            output_norm: None,
        },
    });
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}
