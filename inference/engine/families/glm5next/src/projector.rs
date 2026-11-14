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

use magnitude_artifacts::gguf::{Directory, Element, Scalar, Value};
use magnitude_family_common::{array_elements, projector_type, Tensors};
use magnitude_family_contracts::{
    checked_product, ActivationDType, CellReduction, FamilyError, ImportTransform, MergerStage,
    RmsNorm, RowRange, VisionActivation, VisionAttention, VisionAttentionScale,
    VisionAttentionSpan, VisionBlock, VisionDescription, VisionFeedForward, VisionLinear,
    VisionMerger, VisionNorm, VisionOperator, VisionPreprocessing, VisionResampling,
    VisionResize, VisionStem, VisionUp, WeightDescriptor,
};

/// The projector type the converter writes.
const PROJECTOR_TYPE: &str = "glm5v";

/// The rotary base of the tower (the released vision rotary's default; the
/// converter does not store it).
const ROTARY_BASE: f64 = 10_000.0;

/// Every `clip.*` key the interpretation reads.
const KEYS: &[&str] = &[
    "clip.has_vision_encoder",
    "clip.projector_type",
    "clip.use_silu",
    "clip.vision.projection_dim",
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.block_count",
    "clip.vision.attention.head_count",
    "clip.vision.attention.layer_norm_epsilon",
    "clip.vision.image_mean",
    "clip.vision.image_std",
    "clip.vision.spatial_merge_size",
    "clip.vision.swiglu_clamp",
    "clip.vision.image_min_pixels",
    "clip.vision.image_max_pixels",
];

fn invalid(message: impl Into<String>) -> FamilyError {
    FamilyError::invalid(message)
}

fn product(dimensions: &[u64]) -> Result<u64, FamilyError> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

struct Reader<'a> {
    directory: &'a Directory,
}

impl Reader<'_> {
    fn count(&self, key: &str) -> Result<u64, FamilyError> {
        self.directory
            .value(key)
            .and_then(Value::unsigned)
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid(format!("{key} must be a positive integer")))
    }

    fn number(&self, key: &str) -> Result<f64, FamilyError> {
        match self.directory.value(key) {
            Some(Value::Scalar(Scalar::Float(value))) if value.is_finite() && *value > 0.0 => {
                Ok(*value)
            }
            _ => Err(invalid(format!("{key} must be a positive number"))),
        }
    }

    fn triple(&self, key: &str) -> Result<[f64; 3], FamilyError> {
        let Some(Value::Array(values)) = self.directory.value(key) else {
            return Err(invalid(format!("{key} must contain three numbers")));
        };
        array_elements(key, values)?
            .iter()
            .map(|value| match value {
                Element::Float(value) if value.is_finite() => Some(value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.try_into().ok())
            .ok_or_else(|| invalid(format!("{key} must contain three finite numbers")))
    }
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
) -> Result<VisionDescription, FamilyError> {
    projector_type(directory, &[(PROJECTOR_TYPE, ())], target_width)?;
    if let Some(entry) = directory
        .metadata
        .iter()
        .find(|entry| !entry.name.starts_with("general.") && !KEYS.contains(&entry.name.as_str()))
    {
        return Err(FamilyError::metadata(
            entry.name.clone(),
            format!("unknown projector metadata {:?}", entry.name),
        ));
    }
    let r = Reader { directory };
    // The tower's feed-forward is SiLU.
    if !matches!(
        directory.value("clip.use_silu"),
        Some(Value::Scalar(Scalar::Bool(true)))
    ) {
        return Err(invalid("the projector's feed-forward must be SwiGLU (clip.use_silu)"));
    }
    let depth = r.count("clip.vision.block_count")?;
    let hidden = r.count("clip.vision.embedding_length")?;
    let intermediate = r.count("clip.vision.feed_forward_length")?;
    let heads = r.count("clip.vision.attention.head_count")?;
    let patch = r.count("clip.vision.patch_size")?;
    let merge = r.count("clip.vision.spatial_merge_size")?;
    let output_hidden = r.count("clip.vision.projection_dim")?;
    r.count("clip.vision.image_size")?;
    let min_pixels = r.count("clip.vision.image_min_pixels")?;
    let max_pixels = r.count("clip.vision.image_max_pixels")?;
    let epsilon = r.number("clip.vision.attention.layer_norm_epsilon")?;
    let limit = r.number("clip.vision.swiglu_clamp")?;
    if !hidden.is_multiple_of(heads)
        || !(hidden / heads).is_multiple_of(4)
        || depth > directory.tensors.len() as u64
    {
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

    // One stored QKV projection, imported as three row ranges.
    let qkv_rows = product(&[3, hidden])?;
    let part = |stored: &WeightDescriptor, index: u64| WeightDescriptor {
        name: stored.name.clone(),
        shape: {
            let mut shape = stored.shape.clone();
            shape[0] = hidden;
            shape
        },
        transforms: vec![ImportTransform::Rows(RowRange {
            start: index * hidden,
            rows: hidden,
        })],
    };
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let qkv = weight("attn_qkv.weight", &[qkv_rows, hidden])?;
        let qkv_bias = weight("attn_qkv.bias", &[qkv_rows])?;
        let split = |index| biased(part(&qkv, index), part(&qkv_bias, index));
        blocks.push(VisionBlock {
            operator_norm: rms(weight("ln1.weight", &[hidden])?),
            operator: VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads: heads,
                width,
                query: split(0),
                key: split(1),
                value: split(2),
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
    let description = VisionDescription {
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
            mean: r.triple("clip.vision.image_mean")?,
            std: r.triple("clip.vision.image_std")?,
            channels: 3,
            patch,
            merge,
        },
        stem,
        window: None,
        blocks,
        merger,
    };
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}
