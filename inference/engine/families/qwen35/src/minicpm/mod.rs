//! MiniCPM-V 4.7's 4× SigLIP projector and canvas input interpretation.
pub mod inputs;

use crate::{Error, invalid};
use magnitude_artifacts::gguf::Directory;
use magnitude_family_common::{ProjectorMetadata, Tensors, projector_type};
use magnitude_family_contracts::{
    ActivationDType, CellReduction, MergerStage, PositionSampling, VisionActivation,
    VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock, VisionDescription,
    VisionFeedForward, VisionLinear, VisionMerger, VisionNorm, VisionOperator, VisionPositions,
    VisionPreprocessing, VisionResampling, VisionResize, VisionStem, VisionUp, checked_product,
};

struct Reader<'a> {
    tensors: Tensors<'a>,
    epsilon: f64,
}
impl Reader<'_> {
    fn norm(&mut self, name: &str, width: u64) -> Result<VisionNorm, Error> {
        Ok(VisionNorm::Layer {
            weight: self.tensors.bind(&format!("{name}.weight"), &[width])?,
            bias: self.tensors.bind(&format!("{name}.bias"), &[width])?,
            epsilon: self.epsilon,
        })
    }
    fn linear(&mut self, name: &str, outputs: u64, inputs: u64) -> Result<VisionLinear, Error> {
        Ok(VisionLinear {
            weight: self
                .tensors
                .bind(&format!("{name}.weight"), &[outputs, inputs])?,
            bias: Some(self.tensors.bind(&format!("{name}.bias"), &[outputs])?),
            clamp: None,
        })
    }
}

/// Interpret a MiniCPM-V projector for a target of `target_width`.
pub fn describe_projector(
    directory: &Directory,
    target_width: u64,
) -> Result<Box<VisionDescription>, Error> {
    projector_type(directory, &[("minicpmv4_7", ())], target_width)?;
    let meta = ProjectorMetadata::new(directory);
    meta.ignore("clip.has_vision_encoder");
    meta.ignore("clip.projector_type");
    meta.ignore("clip.vision.wa_layer_indexes");
    let epsilon = meta.positive_number("clip.vision.attention.layer_norm_epsilon")?;
    let mut r = Reader {
        tensors: Tensors::new(directory),
        epsilon,
    };
    if meta.integer("clip.vision.projector.scale_factor")? != 2 {
        return Err(invalid(
            "MiniCPM-V 16x intra-tower compression is not supported by this projector",
        ));
    }
    if !meta.bool("clip.use_gelu")? {
        return Err(invalid("MiniCPM-V requires the published GELU tower"));
    }
    let hidden = meta.integer("clip.vision.embedding_length")?;
    let output_hidden = meta.integer("clip.vision.projection_dim")?;
    let intermediate = meta.integer("clip.vision.feed_forward_length")?;
    let depth = meta.bounded_block_count("clip.vision.block_count")?;
    let heads = meta.integer("clip.vision.attention.head_count")?;
    let patch = meta.integer("clip.vision.patch_size")?;
    if hidden % heads != 0 || (hidden / heads) % 4 != 0 {
        return Err(invalid("invalid MiniCPM-V tower geometry"));
    }
    let [positions, position_hidden] = r.tensors.shape("v.position_embd.weight")? else {
        return Err(invalid("position embedding must be a matrix"));
    };
    if *position_hidden != hidden {
        return Err(invalid("position embedding width differs from tower"));
    }
    let side = (*positions as f64).sqrt() as u64;
    if side.checked_mul(side) != Some(*positions) {
        return Err(invalid("position embedding must have a square bucket grid"));
    }
    let positions = VisionPositions {
        table: r
            .tensors
            .bind("v.position_embd.weight", &[side * side, hidden])?,
        sampling: PositionSampling::Buckets { side },
    };
    let stem = VisionStem::Patch {
        frames: vec![
            r.tensors
                .bind("v.patch_embd.weight", &[hidden, 3, patch, patch])?,
        ],
        bias: Some(r.tensors.bind("v.patch_embd.bias", &[hidden])?),
        positions: Some(positions),
        norm: None,
        registers: None,
    };
    let mut blocks = Vec::with_capacity(depth as usize);
    for layer in 0..depth {
        let p = format!("v.blk.{layer}");
        blocks.push(VisionBlock {
            operator_norm: r.norm(&format!("{p}.ln1"), hidden)?,
            operator: VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads: heads,
                width: hidden / heads,
                query: r.linear(&format!("{p}.attn_q"), hidden, hidden)?,
                key: r.linear(&format!("{p}.attn_k"), hidden, hidden)?,
                value: r.linear(&format!("{p}.attn_v"), hidden, hidden)?,
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: None,
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: r.linear(&format!("{p}.attn_out"), hidden, hidden)?,
            }),
            operator_post_norm: None,
            feedforward_norm: r.norm(&format!("{p}.ln2"), hidden)?,
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(r.linear(&format!("{p}.ffn_up"), intermediate, hidden)?),
                activation: VisionActivation::GeluTanh,
                down: r.linear(&format!("{p}.ffn_down"), hidden, intermediate)?,
            },
            feedforward_post_norm: None,
        });
    }
    let joined = checked_product(&[4, hidden]).map_err(|e| invalid(e.to_string()))?;
    let merger = VisionMerger {
        norm: Some(r.norm("v.post_ln", hidden)?),
        reduction: CellReduction::Concatenate,
        standardize: None,
        projection_norm: Some(r.norm("mm.input_norm", joined)?),
        stages: vec![
            MergerStage {
                norm: None,
                linear: r.linear("mm.up", joined, joined)?,
                activation: Some(VisionActivation::GeluErf),
            },
            MergerStage {
                norm: None,
                linear: r.linear("mm.down", output_hidden, joined)?,
                activation: None,
            },
        ],
        feed_forward: None,
        output_norm: None,
    };
    // The released graph reads the four boundary tokens from the target's
    // embedding table. Some GGUF exports also carry this unused helper table;
    // validate its exact geometry rather than assigning it an encoder role.
    if r.tensors.contains("v.tok_embd_sep") {
        r.tensors.bind("v.tok_embd_sep", &[4, output_hidden])?;
    }
    let preprocessing = VisionPreprocessing {
        resize: VisionResize::Slices {
            resolution: meta.integer("clip.vision.image_size")?,
            max_slices: meta.integer("clip.vision.max_slice_nums")?,
        },
        resampling: VisionResampling::Bicubic,
        mean: meta.f64_array("clip.vision.image_mean")?,
        std: meta.f64_array("clip.vision.image_std")?,
        channels: 3,
        patch,
        merge: 2,
    };
    meta.finish()?;
    r.tensors.finish().map_err(Error::from)?;
    let description = Box::new(VisionDescription {
        activation_dtype: ActivationDType::F16,
        hidden,
        output_hidden,
        preprocessing,
        stem,
        window: None,
        blocks,
        merger,
    });
    description.validate().map_err(|e| invalid(e.to_string()))?;
    Ok(description)
}
