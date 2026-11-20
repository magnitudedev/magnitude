//! The Muse Glimmer projector (`clip.projector_type = muse-glimmer`): a
//! LayerNorm ViT over 14-pixel patches whose sparse layers attend within
//! 32 × 32-patch windows, a channel-outer 2 × 2 pixel shuffle and a
//! three-stage erf-GELU adapter to the decoder width.

use crate::{invalid, product, Error};
use magnitude_artifacts::gguf::Directory;
use magnitude_family_common::{projector_type, rotary, ProjectorMetadata, Tensors};
use magnitude_family_contracts::{
    ActivationDType, CellReduction, ImportTransform, MergerStage, PositionSampling,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm, VisionOperator,
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

/// Interpret the Muse Glimmer projector. `output_hidden` is the decoder
/// width; `feature_epsilon` the decoder's RMS epsilon, which the released
/// model's weightless `perception_emb_norm` of every feature row uses.
pub fn describe_projector(
    directory: &Directory,
    output_hidden: u64,
    feature_epsilon: f64,
) -> Result<Box<VisionDescription>, Error> {
    projector_type(directory, &[("muse-glimmer", ())], output_hidden)?;
    let meta = ProjectorMetadata::open(directory);
    meta.ignore("clip.vision.image_size");
    meta.invariant_flag("clip.use_gelu", true, "true (GELU feed-forward)")?;
    // Declared, and the target's (`projector_type`).
    meta.integer("clip.vision.projection_dim")?;
    let hidden = meta.integer("clip.vision.embedding_length")?;
    let intermediate = meta.integer("clip.vision.feed_forward_length")?;
    let depth = meta.bounded_block_count("clip.vision.block_count")?;
    let heads = meta.integer("clip.vision.attention.head_count")?;
    let patch = meta.integer("clip.vision.patch_size")?;
    let merge = meta.integer("clip.vision.spatial_merge_size")?;
    let epsilon = meta.positive_number("clip.vision.attention.layer_norm_epsilon")?;
    if hidden % heads != 0 || (hidden / heads) % 4 != 0 {
        return Err(invalid("projector heads do not split into 2-D rotary heads"));
    }
    let width = hidden / heads;

    let mut t = Tensors::new(directory);
    // The window side is the position table's side: llama.cpp and the
    // released model both tie them.
    let (table_rows, table_cols) = t.matrix_or_batched_shape("v.position_embd.weight")?;
    let side = (table_rows as f64).sqrt() as u64;
    if side * side != table_rows || table_cols != hidden {
        return Err(invalid("projector position table is not square"));
    }
    let table = t.bind_matrix_or_batched("v.position_embd.weight", table_rows, hidden)?;
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
            operator_norm: layer_norm(&mut t, &format!("{p}ln1"))?,
            operator: VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads: heads,
                width,
                query: linear(&mut t, &format!("{p}attn_q"), hidden, hidden, rotary.clone())?,
                key: linear(&mut t, &format!("{p}attn_k"), hidden, hidden, rotary.clone())?,
                value: linear(&mut t, &format!("{p}attn_v"), hidden, hidden, Vec::new())?,
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: Some(ROTARY_BASE),
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: if global {
                    VisionAttentionSpan::Full
                } else {
                    VisionAttentionSpan::Window
                },
                output: linear(&mut t, &format!("{p}attn_out"), hidden, hidden, Vec::new())?,
            }),
            operator_post_norm: None,
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
            norm: None,
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
    let description = Box::new(VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::CellBudget {
                max_cells: MAX_CELLS,
            },
            resampling: VisionResampling::Lanczos,
            mean: meta.f64_array("clip.vision.image_mean")?,
            std: meta.f64_array("clip.vision.image_std")?,
            channels: 3,
            patch,
            merge,
        },
        stem: VisionStem::Patch {
            frames: vec![frame],
            bias: None,
            positions: Some(VisionPositions {
                table,
                sampling: PositionSampling::PixelCenters { side },
            }),
            norm: Some(pre_norm),
            registers: None,
        },
        window: Some(side),
        blocks,
        merger: VisionMerger {
            norm: Some(post_norm),
            reduction: CellReduction::Interleave,
            standardize: None,
            projection_norm: None,
            stages,
            feed_forward: None,
            output_norm: Some(VisionNorm::Rms {
                weight: None,
                epsilon: feature_epsilon,
            }),
        },
    });
    meta.finish()?;
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}
