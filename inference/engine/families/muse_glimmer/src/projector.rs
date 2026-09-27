//! The Muse Glimmer projector (`clip.projector_type = muse-glimmer`): a
//! LayerNorm ViT over 14-pixel patches whose sparse layers attend within
//! 32 × 32-patch windows, a channel-outer 2 × 2 pixel shuffle and a
//! three-stage erf-GELU adapter to the decoder width.

use crate::{invalid, product, Error};
use magnitude_artifacts::gguf::{Directory, Scalar, Value};
use magnitude_family_common::{rotary, Tensors};
use magnitude_family_contracts::{
    ActivationDType, CellReduction, ImportTransform, MergerStage, PositionSampling,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm,
    VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem, VisionUp,
};

/// The released vision configuration's facts the header does not record: the
/// rotary base (`rope_parameters.rope_theta`), the period of the global
/// layers (every fourth and the last; the others attend within windows), the
/// image processor's cell budget (`max_image_tokens`) and its Lanczos
/// resampling, and the erf GELU of both the tower (`hidden_act = gelu`) and
/// the adapter (`projector_hidden_act = gelu`).
const ROTARY_BASE: f64 = 10_000.0;
const GLOBAL_PERIOD: u64 = 4;
const MAX_CELLS: u64 = 4096;
const ACTIVATION: VisionActivation = VisionActivation::GeluErf;

/// Every `clip.` key the interpretation reads; any other is rejected.
const KEYS: &[&str] = &[
    "clip.has_vision_encoder",
    "clip.projector_type",
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
];

fn unsigned(directory: &Directory, key: &str) -> Result<u64, Error> {
    directory
        .value(key)
        .and_then(Value::unsigned)
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{key} must be a positive integer")))
}

fn number(directory: &Directory, key: &str) -> Result<f64, Error> {
    match directory.value(key) {
        Some(Value::Scalar(Scalar::Float(value))) if value.is_finite() && *value > 0.0 => {
            Ok(*value)
        }
        _ => Err(invalid(format!("{key} must be a positive number"))),
    }
}

fn triple(directory: &Directory, key: &str) -> Result<[f64; 3], Error> {
    let values = match directory.value(key) {
        Some(Value::Array(values)) if values.len() == 3 => values
            .iter()
            .map(|value| match value {
                Scalar::Float(value) if value.is_finite() => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>(),
        _ => None,
    }
    .ok_or_else(|| invalid(format!("{key} must hold three numbers")))?;
    Ok([values[0], values[1], values[2]])
}

/// Interpret the Muse Glimmer projector. `output_hidden` is the decoder
/// width; `feature_epsilon` the decoder's RMS epsilon, which the released
/// model's weightless `perception_emb_norm` of every feature row uses.
pub fn describe_projector(
    directory: &Directory,
    output_hidden: u64,
    feature_epsilon: f64,
) -> Result<VisionDescription, Error> {
    if directory.value("general.architecture").and_then(Value::string) != Some("clip")
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory.value("clip.has_vision_encoder")
            != Some(&Value::Scalar(Scalar::Bool(true)))
        || directory.value("clip.projector_type").and_then(Value::string)
            != Some("muse-glimmer")
    {
        return Err(invalid("unsupported Muse Glimmer projector identity"));
    }
    if let Some(entry) = directory.metadata.iter().find(|entry| {
        !entry.name.starts_with("general.") && !KEYS.contains(&entry.name.as_str())
    }) {
        return Err(invalid(format!("unknown metadata {:?}", entry.name)));
    }
    if unsigned(directory, "clip.vision.projection_dim")? != output_hidden {
        return Err(invalid("projector width differs from the decoder width"));
    }
    let hidden = unsigned(directory, "clip.vision.embedding_length")?;
    let intermediate = unsigned(directory, "clip.vision.feed_forward_length")?;
    let depth = unsigned(directory, "clip.vision.block_count")?;
    let heads = unsigned(directory, "clip.vision.attention.head_count")?;
    let patch = unsigned(directory, "clip.vision.patch_size")?;
    let merge = unsigned(directory, "clip.vision.spatial_merge_size")?;
    let epsilon = number(directory, "clip.vision.attention.layer_norm_epsilon")?;
    if hidden % heads != 0 || (hidden / heads) % 4 != 0 {
        return Err(invalid("projector heads do not split into 2-D rotary heads"));
    }
    let width = hidden / heads;
    if depth > directory.tensors.len() as u64 {
        return Err(invalid("projector block count exceeds its stored weights"));
    }

    let mut t = Tensors::new(directory);
    // The window side is the position table's side: llama.cpp and the
    // released model both tie them.
    let table_rows = t.rows("v.position_embd.weight")?;
    let side = (table_rows as f64).sqrt() as u64;
    if side * side != table_rows {
        return Err(invalid("projector position table is not square"));
    }
    let table = t.bind("v.position_embd.weight", &[table_rows, hidden])?;
    let layer_norm = |t: &mut Tensors, stem: &str| -> Result<VisionNorm, Error> {
        Ok(VisionNorm::Layer {
            weight: t.bind(&format!("{stem}.weight"), &[hidden])?,
            bias: t.bind(&format!("{stem}.bias"), &[hidden])?,
            epsilon,
        })
    };
    let linear = |t: &mut Tensors, stem: &str, outputs, inputs, transforms: Vec<ImportTransform>| {
        Ok::<_, Error>(VisionLinear {
            weight: t.bind_transformed(
                &format!("{stem}.weight"),
                &[outputs, inputs],
                transforms.clone(),
            )?,
            bias: Some(t.bind_transformed(&format!("{stem}.bias"), &[outputs], transforms)?),
            clamp: None,
        })
    };
    // The converter stores query and key rows in llama.cpp's adjacent-pair
    // layout (`_unpermute_for_rope`: stored row `2j + h` is released row
    // `h · W/2 + j`); the released layout is the vision contract's (column
    // `i < W/2` pairs with `i + W/2`).
    let rotary = vec![rotary::half_split_rows(width, width)];
    let pre_norm = layer_norm(&mut t, "v.pre_ln")?;
    let post_norm = layer_norm(&mut t, "v.post_ln")?;
    let mut blocks = Vec::new();
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let global = index + 1 == depth || (index + 1) % GLOBAL_PERIOD == 0;
        blocks.push(VisionBlock {
            attention_norm: layer_norm(&mut t, &format!("{p}ln1"))?,
            attention: VisionAttention {
                heads,
                width,
                query: linear(&mut t, &format!("{p}attn_q"), hidden, hidden, rotary.clone())?,
                key: linear(&mut t, &format!("{p}attn_k"), hidden, hidden, rotary.clone())?,
                value: linear(&mut t, &format!("{p}attn_v"), hidden, hidden, Vec::new())?,
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: ROTARY_BASE,
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: if global {
                    VisionAttentionSpan::Full
                } else {
                    VisionAttentionSpan::Window
                },
                output: linear(&mut t, &format!("{p}attn_out"), hidden, hidden, Vec::new())?,
            },
            attention_post_norm: None,
            feedforward_norm: layer_norm(&mut t, &format!("{p}ln2"))?,
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(linear(
                    &mut t,
                    &format!("{p}ffn_up"),
                    intermediate,
                    hidden,
                    Vec::new(),
                )?),
                activation: ACTIVATION,
                down: linear(&mut t, &format!("{p}ffn_down"), hidden, intermediate, Vec::new())?,
            },
            feedforward_post_norm: None,
        });
    }
    let shuffled = product(&[hidden, merge, merge])?;
    let adapter = t.rows("mm.0.weight")?;
    let mut stage = |name: &str, outputs, inputs, activation| {
        Ok::<_, Error>(MergerStage {
            linear: VisionLinear {
                weight: t.bind(name, &[outputs, inputs])?,
                bias: None,
                clamp: None,
            },
            activation,
        })
    };
    let stages = vec![
        stage("mm.0.weight", adapter, shuffled, Some(ACTIVATION))?,
        stage("mm.1.weight", adapter, adapter, Some(ACTIVATION))?,
        stage("mm.2.weight", output_hidden, adapter, None)?,
    ];
    // The converter sums the released model's two temporal patch slabs into
    // one frame (a still image repeats its frame).
    let frame = t.bind("v.patch_embd.weight", &[hidden, 3, patch, patch])?;
    t.finish()?;
    let description = VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::CellBudget {
                max_cells: MAX_CELLS,
            },
            resampling: VisionResampling::Lanczos,
            mean: triple(directory, "clip.vision.image_mean")?,
            std: triple(directory, "clip.vision.image_std")?,
            channels: 3,
            patch,
            merge,
        },
        stem: VisionStem::Patch {
            frames: vec![frame],
            bias: None,
            positions: VisionPositions {
                table,
                sampling: PositionSampling::PixelCenters { side },
            },
            norm: Some(pre_norm),
        },
        window: Some(side),
        blocks,
        merger: VisionMerger {
            norm: Some(post_norm),
            reduction: CellReduction::Interleave,
            standardize: None,
            projection_norm: None,
            stages,
            output_norm: Some(VisionNorm::Rms {
                weight: None,
                epsilon: feature_epsilon,
            }),
        },
    };
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}
