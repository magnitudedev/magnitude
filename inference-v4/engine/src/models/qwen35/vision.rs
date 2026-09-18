//! Device-free MLX image-tower geometry and weight roles. Encoder execution is
//! separate from interpretation; text-only loading need not inspect these roles.
use super::preparation::ImageGeometry;
use crate::{
    inputs::TokenId,
    weights::{descriptor::WeightDescriptor, mlx::MlxArtifact},
};
use serde::Deserialize;

#[derive(Clone, Debug)]
pub struct Geometry {
    pub image: ImageGeometry,
    pub hidden: usize,
    pub intermediate: usize,
    pub output: usize,
    pub heads: usize,
    pub depth: usize,
    pub table_side: usize,
}
#[derive(Clone, Debug)]
pub struct AffineWeights {
    pub weight: WeightDescriptor,
    pub bias: WeightDescriptor,
}
#[derive(Clone, Debug)]
pub struct BlockWeights {
    pub norm1: AffineWeights,
    pub qkv: AffineWeights,
    pub projection: AffineWeights,
    pub norm2: AffineWeights,
    pub up: AffineWeights,
    pub down: AffineWeights,
}
#[derive(Clone, Debug)]
pub struct Description {
    pub geometry: Geometry,
    pub patch: AffineWeights,
    pub positions: WeightDescriptor,
    pub blocks: Vec<BlockWeights>,
    pub merger_norm: AffineWeights,
    pub merger_up: AffineWeights,
    pub merger_down: AffineWeights,
}
#[derive(Deserialize)]
struct VisionConfig {
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    out_hidden_size: usize,
    num_heads: usize,
    depth: usize,
    num_position_embeddings: usize,
    hidden_act: String,
    deepstack_visual_indexes: Vec<usize>,
}
#[derive(Deserialize)]
struct TextConfig {
    hidden_size: usize,
}
#[derive(Deserialize)]
struct Config {
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    vision_config: VisionConfig,
    text_config: TextConfig,
}
pub fn describe(artifact: &MlxArtifact) -> Result<Description, String> {
    let config: Config = serde_json::from_value(artifact.config().clone())
        .map_err(|e| format!("vision configuration: {e}"))?;
    let v = config.vision_config;
    if !v.deepstack_visual_indexes.is_empty() || v.hidden_act != "gelu_pytorch_tanh" {
        return Err("unsupported Qwen image tower".into());
    }
    if [
        v.in_channels,
        v.temporal_patch_size,
        v.patch_size,
        v.spatial_merge_size,
        v.hidden_size,
        v.intermediate_size,
        v.out_hidden_size,
        v.num_heads,
        v.depth,
        v.num_position_embeddings,
    ]
    .iter()
    .any(|&n| n == 0 || n > i32::MAX as usize)
    {
        return Err("invalid vision dimensions".into());
    }
    let table_side = v.num_position_embeddings.isqrt();
    if table_side * table_side != v.num_position_embeddings {
        return Err("vision position table must be square".into());
    }
    if v.num_heads
        .checked_mul(4)
        .is_none_or(|n| v.hidden_size % n != 0)
    {
        return Err("vision heads require complete rotary quarter-pairs".into());
    }
    if v.out_hidden_size != config.text_config.hidden_size {
        return Err("image projection differs from decoder width".into());
    }
    if v.depth > artifact.tensors().len() / 12 {
        return Err("vision depth exceeds available weight roles".into());
    }
    let image = ImageGeometry {
        channels: v.in_channels,
        temporal_patch: v.temporal_patch_size,
        patch: v.patch_size,
        merge: v.spatial_merge_size,
        image_token: TokenId(config.image_token_id),
        start_token: TokenId(config.vision_start_token_id),
        end_token: TokenId(config.vision_end_token_id),
    };
    let width = image.patch_width()?;
    let merged = v
        .hidden_size
        .checked_mul(image.merge)
        .and_then(|n| n.checked_mul(image.merge))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or("merged vision width exceeds index domain")?;
    let qkv_width = v
        .hidden_size
        .checked_mul(3)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or("vision projection exceeds index domain")?;
    let patch = artifact
        .tensors()
        .get("vision_tower.patch_embed.proj.weight")
        .ok_or("missing vision patch kernel")?;
    if patch.shape
        != [
            v.hidden_size as u64,
            image.temporal_patch as u64,
            image.patch as u64,
            image.patch as u64,
            image.channels as u64,
        ]
    {
        return Err(
            "MLX vision patch kernel requires output/time/height/width/channel order".into(),
        );
    }
    let weight = |name: &str, shape: &[usize]| {
        artifact
            .descriptor(
                &format!("vision_tower.{name}"),
                &shape.iter().map(|&n| n as u64).collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())
    };
    let affine =
        |name: &str, output: usize, input: Option<usize>| -> Result<AffineWeights, String> {
            let shape = match input {
                Some(input) => vec![output, input],
                None => vec![output],
            };
            Ok(AffineWeights {
                weight: weight(&format!("{name}.weight"), &shape)?,
                bias: weight(&format!("{name}.bias"), &[output])?,
            })
        };
    let blocks = (0..v.depth)
        .map(|i| -> Result<BlockWeights, String> {
            Ok(BlockWeights {
                norm1: affine(&format!("blocks.{i}.norm1"), v.hidden_size, None)?,
                qkv: affine(
                    &format!("blocks.{i}.attn.qkv"),
                    qkv_width,
                    Some(v.hidden_size),
                )?,
                projection: affine(
                    &format!("blocks.{i}.attn.proj"),
                    v.hidden_size,
                    Some(v.hidden_size),
                )?,
                norm2: affine(&format!("blocks.{i}.norm2"), v.hidden_size, None)?,
                up: affine(
                    &format!("blocks.{i}.mlp.linear_fc1"),
                    v.intermediate_size,
                    Some(v.hidden_size),
                )?,
                down: affine(
                    &format!("blocks.{i}.mlp.linear_fc2"),
                    v.hidden_size,
                    Some(v.intermediate_size),
                )?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Description {
        patch: affine("patch_embed.proj", v.hidden_size, Some(width))?,
        positions: weight(
            "pos_embed.weight",
            &[v.num_position_embeddings, v.hidden_size],
        )?,
        merger_norm: affine("merger.norm", v.hidden_size, None)?,
        merger_up: affine("merger.linear_fc1", merged, Some(merged))?,
        merger_down: affine("merger.linear_fc2", v.out_hidden_size, Some(merged))?,
        blocks,
        geometry: Geometry {
            image,
            hidden: v.hidden_size,
            intermediate: v.intermediate_size,
            output: v.out_hidden_size,
            heads: v.num_heads,
            depth: v.depth,
            table_side,
        },
    })
}
