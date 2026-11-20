//! The GLM-5.3-Flash projector (`glm5v`): a 24-block ViT over 14-pixel
//! patches, and a merger of the tower's 2 × 2 cells.
//!
//! The tower is the Qwen-style one: a patch convolution over two copies of
//! the image (the stored temporal pair of kernels, with a bias), no position
//! table, 2-D rotary attention over a fused `qkv` projection with RMS norms of
//! each query and key head, and a SwiGLU feed-forward whose gate and up are
//! clamped at the metadata's `swiglu_clamp`. Every projection carries a bias.
//! Blocks are pre-norm RMS; a post RMS norm closes the tower.
//!
//! The merger is the released `Glm5NextVisionPatchMerger` after the
//! `downsample` convolution. The convolution (kernel and stride 2) is a
//! linear over each cell's tower rows interleaved channel by channel, which
//! is how its kernel `[out, in, 2, 2]` reads flattened
//! ([`ImportTransform::MergeTrailingAxes`]). Then `proj`, a layer norm and
//! the erf GELU, and a clamped SwiGLU back to the decoder's width, all
//! without bias.
//!
//! An image is resized into a canvas of whole cells and padded
//! ([`VisionResize::PaddedCanvas`]) within the pixel bounds the metadata
//! states.

use magnitude_artifacts::gguf::Directory;
use magnitude_family_common::{ProjectorMetadata, projector_type, Tensors};
use magnitude_family_contracts::{
    checked_product, ActivationDType, CellReduction, FamilyError, ImportTransform, MergerStage,
    RmsNorm, VisionActivation, VisionAttention, VisionAttentionScale,
    VisionAttentionSpan, VisionBlock, VisionDescription, VisionFeedForward, VisionLinear,
    VisionMerger, VisionNorm, VisionOperator, VisionPreprocessing, VisionResampling,
    VisionResize, VisionStem, VisionUp, WeightDescriptor,
};

/// The projector type the converter writes.
const PROJECTOR_TYPE: &str = "glm5v";

/// The rotary base of the tower (the released vision rotary's default; the
/// converter does not store it).
const ROTARY_BASE: f64 = 10_000.0;

fn invalid(message: impl Into<String>) -> FamilyError {
    FamilyError::invalid(message)
}

fn product(dimensions: &[u64]) -> Result<u64, FamilyError> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

/// The merger's LayerNorm epsilon. The released model builds that norm as
/// `LayerNorm(out_hidden_size)` with PyTorch's default epsilon, apart from
/// the RMS norms' `rms_norm_eps` (the file's `layer_norm_epsilon`), so the
/// file states no epsilon for it.
const POST_NORM_EPSILON: f64 = 1e-5;

/// Interpret a GLM-5.3-Flash projector inventory for a target of
/// `target_width` without importing it.
pub fn describe_projector(
    directory: &Directory,
    target_width: u64,
) -> Result<Box<VisionDescription>, FamilyError> {
    projector_type(directory, &[(PROJECTOR_TYPE, ())], target_width)?;
    let meta = ProjectorMetadata::new(directory);
    meta.ignore("clip.has_vision_encoder");
    meta.ignore("clip.projector_type");
    // The tower's feed-forward is SiLU.
    if !meta.bool("clip.use_silu")? {
        return Err(invalid("the projector's feed-forward must be SwiGLU (clip.use_silu)"));
    }
    let depth = meta.bounded_block_count("clip.vision.block_count")?;
    let hidden = meta.integer("clip.vision.embedding_length")?;
    let intermediate = meta.integer("clip.vision.feed_forward_length")?;
    let heads = meta.integer("clip.vision.attention.head_count")?;
    let patch = meta.integer("clip.vision.patch_size")?;
    let merge = meta.integer("clip.vision.spatial_merge_size")?;
    let output_hidden = meta.integer("clip.vision.projection_dim")?;
    meta.integer("clip.vision.image_size")?;
    let min_pixels = meta.integer("clip.vision.image_min_pixels")?;
    let max_pixels = meta.integer("clip.vision.image_max_pixels")?;
    let epsilon = meta.positive_number("clip.vision.attention.layer_norm_epsilon")?;
    let limit = meta.positive_number("clip.vision.swiglu_clamp")?;
    if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(4) {
        return Err(invalid("projector heads do not split into 2-D rotary heads"));
    }
    let width = hidden / heads;

    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let rms = |weight: WeightDescriptor| VisionNorm::Rms {
        weight: Some(weight),
        epsilon,
    };
    let biased = |weight: WeightDescriptor, bias: WeightDescriptor| VisionLinear {
        weight,
        bias: Some(bias),
        clamp: None,
    };
    let unbiased = |weight: WeightDescriptor| VisionLinear {
        weight,
        bias: None,
        clamp: None,
    };

    let frames = ["v.patch_embd.weight", "v.patch_embd.weight.1"]
        .into_iter()
        .map(|name| tensor(name, &[hidden, 3, patch, patch]))
        .collect::<Result<Vec<_>, _>>()?;
    let stem = VisionStem::Patch {
        frames,
        bias: Some(tensor("v.patch_embd.bias", &[hidden])?),
        positions: None,
        norm: None,
        registers: None,
    };

    // One stored QKV projection, imported as three row ranges (preserving any companion scale).
    let qkv_rows = product(&[3, hidden])?;
    let part = |stored: &WeightDescriptor, index: u64| {
        stored
            .slice_rows(index * hidden, hidden)
            .map_err(|error| FamilyError::invalid(error.to_string()))
    };
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let qkv = weight("attn_qkv.weight", &[qkv_rows, hidden])?;
        let qkv_bias = weight("attn_qkv.bias", &[qkv_rows])?;
        let split = |index| -> Result<VisionLinear, FamilyError> {
            Ok(biased(part(&qkv, index)?, part(&qkv_bias, index)?))
        };
        blocks.push(VisionBlock {
            operator_norm: rms(weight("ln1.weight", &[hidden])?),
            operator: VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads: heads,
                width,
                query: split(0)?,
                key: split(1)?,
                value: split(2)?,
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
                output: biased(
                    weight("attn_out.weight", &[hidden, hidden])?,
                    weight("attn_out.bias", &[hidden])?,
                ),
            }),
            operator_post_norm: None,
            feedforward_norm: rms(weight("ln2.weight", &[hidden])?),
            feedforward: VisionFeedForward {
                up: VisionUp::Gated {
                    gate: biased(
                        weight("ffn_gate.weight", &[intermediate, hidden])?,
                        weight("ffn_gate.bias", &[intermediate])?,
                    ),
                    up: biased(
                        weight("ffn_up.weight", &[intermediate, hidden])?,
                        weight("ffn_up.bias", &[intermediate])?,
                    ),
                    limit: Some(limit),
                },
                activation: VisionActivation::Silu,
                down: biased(
                    weight("ffn_down.weight", &[hidden, intermediate])?,
                    weight("ffn_down.bias", &[hidden])?,
                ),
            },
            feedforward_post_norm: None,
        });
    }

    // The downsample convolution `[out, in, merge, merge]` as the matrix of
    // its flattened windows, `in`-major: the cell's rows interleaved channel
    // by channel.
    let downsample = tensors.bind_transformed(
        "mm.patch_merger.weight",
        &[output_hidden, hidden, merge, merge],
        vec![ImportTransform::MergeTrailingAxes],
    )?;
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let projection = tensor("mm.model.fc.weight", &[output_hidden, output_hidden])?;
    let merger_intermediate = tensors.rows("mm.gate.weight")?;
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let merger = VisionMerger {
        norm: Some(rms(tensor("v.post_ln.weight", &[hidden])?)),
        reduction: CellReduction::Interleave,
        standardize: None,
        projection_norm: None,
        stages: vec![
            MergerStage {
                linear: biased(downsample, tensor("mm.patch_merger.bias", &[output_hidden])?),
                norm: None,
                activation: None,
            },
            MergerStage {
                linear: unbiased(projection),
                norm: Some(VisionNorm::Layer {
                    weight: tensor("mm.post_norm.weight", &[output_hidden])?,
                    bias: tensor("mm.post_norm.bias", &[output_hidden])?,
                    epsilon: POST_NORM_EPSILON,
                }),
                activation: Some(VisionActivation::GeluErf),
            },
        ],
        feed_forward: Some(VisionFeedForward {
            up: VisionUp::Gated {
                gate: unbiased(tensor(
                    "mm.gate.weight",
                    &[merger_intermediate, output_hidden],
                )?),
                up: unbiased(tensor("mm.up.weight", &[merger_intermediate, output_hidden])?),
                limit: Some(limit),
            },
            activation: VisionActivation::Silu,
            down: unbiased(tensor(
                "mm.down.weight",
                &[output_hidden, merger_intermediate],
            )?),
        }),
        output_norm: None,
    };
    tensors.finish()?;
    let description = Box::new(VisionDescription {
        // F16 intermediates around an F32 residual stream, as the Qwen3-VL
        // tower the vision kernels were measured on: the clamps bound the
        // feed-forward products, and the norms bound the rest.
        activation_dtype: ActivationDType::F16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PaddedCanvas {
                min_pixels,
                max_pixels,
            },
            resampling: VisionResampling::Bicubic,
            mean: meta.f64_array("clip.vision.image_mean")?,
            std: meta.f64_array("clip.vision.image_std")?,
            channels: 3,
            patch,
            merge,
        },
        stem,
        window: None,
        blocks,
        merger,
    });
    meta.finish()?;
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}
