//! The Inkling 2 vision projector (`inkling2v`): a 23-layer ViT with 2D axial
//! RoPE, bilinearly sampled 32×32 position embeddings, pre-LayerNorm,
//! LayerScale (`ls1`/`ls2`), and a 2-layer MLP spatial merger.

use magnitude_artifacts::gguf::Directory;
use magnitude_family_common::{ProjectorMetadata, Tensors, projector_type, rotary};
use magnitude_family_contracts::{
    ActivationDType, CellReduction, FamilyError, ImportTransform, MergerStage, PositionSampling,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionLinear, VisionMerger, VisionNorm, VisionOperator,
    VisionPositions, VisionPreprocessing, VisionResampling, VisionResize, VisionStem, VisionUp,
    WeightDescriptor, checked_product,
};

const PROJECTOR_TYPE: &str = "inkling2v";
const ROTARY_BASE: f64 = 10_000.0;

fn invalid(message: impl Into<String>) -> FamilyError {
    FamilyError::invalid(message)
}

fn product(dimensions: &[u64]) -> Result<u64, FamilyError> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

/// Binds an optional single-token class embedding or class position tensor,
/// accepting `[hidden]` or `[1, hidden]` consistently.
fn bind_optional_class_vector(
    tensors: &mut Tensors<'_>,
    name: &str,
    hidden: u64,
) -> Result<Option<WeightDescriptor>, FamilyError> {
    if !tensors.contains(name) {
        return Ok(None);
    }
    let shape = tensors.shape(name)?.to_vec();
    match shape.as_slice() {
        [cols] | [1, cols] if *cols == hidden => Ok(Some(tensors.bind(name, &[hidden])?)),
        _ => Err(invalid(format!(
            "inkling2v {name} has unexpected shape {shape:?}, expected [{hidden}] or [1, {hidden}]"
        ))),
    }
}

/// Interpret an Inkling 2 vision projector (`inkling2v`) for a decoder of
/// `target_width`.
pub fn describe_projector(
    directory: &Directory,
    target_width: u64,
) -> Result<Box<VisionDescription>, FamilyError> {
    projector_type(directory, &[(PROJECTOR_TYPE, ())], target_width)?;
    let meta = ProjectorMetadata::new(directory);
    meta.ignore("clip.has_vision_encoder");
    meta.ignore("clip.projector_type");
    meta.ignore("clip.vision.projector_type");
    let depth = meta.bounded_block_count("clip.vision.block_count")?;
    let hidden = meta.integer("clip.vision.embedding_length")?;
    let intermediate = meta.integer("clip.vision.feed_forward_length")?;
    let heads = meta.integer("clip.vision.attention.head_count")?;
    let patch = meta.integer("clip.vision.patch_size")?;
    let merge = meta.integer("clip.vision.spatial_merge_size")?;
    let output_hidden = meta.integer("clip.vision.projection_dim")?;
    let image_size = meta.integer("clip.vision.image_size")?;
    let epsilon = meta.positive_number("clip.vision.attention.layer_norm_epsilon")?;
    let rotary_base = meta
        .optional_positive_number("clip.vision.rope.freq_base")?
        .unwrap_or(ROTARY_BASE);
    if let Some(use_gelu) = meta.optional_bool("clip.use_gelu")? {
        if !use_gelu {
            return Err(invalid("inkling2v clip.use_gelu must be true"));
        }
    }
    if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(4) {
        return Err(invalid(
            "inkling2v projector heads do not split into 2-D rotary heads",
        ));
    }
    if !image_size.is_multiple_of(patch) {
        return Err(invalid(
            "inkling2v image_size is not a multiple of patch_size",
        ));
    }
    let width = hidden / heads;
    let side = image_size / patch;
    if let Some(grid_size) = meta.optional_integer("clip.vision.position_grid_size")? {
        if grid_size != side {
            return Err(invalid(format!(
                "inkling2v clip.vision.position_grid_size ({grid_size}) disagrees with image_size/patch_size ({side})"
            )));
        }
    }
    if let Some(scale_factor) = meta.optional_integer("clip.vision.projector.scale_factor")? {
        if scale_factor != merge {
            return Err(invalid(format!(
                "inkling2v clip.vision.projector.scale_factor ({scale_factor}) disagrees with spatial_merge_size ({merge})"
            )));
        }
    }
    let max_patches = product(&[side, side])?;

    let mut tensors = Tensors::new(directory);
    let layer_norm = |tensors: &mut Tensors, stem: &str| -> Result<VisionNorm, FamilyError> {
        Ok(VisionNorm::Layer {
            weight: tensors.bind(&format!("{stem}.weight"), &[hidden])?,
            bias: tensors.bind(&format!("{stem}.bias"), &[hidden])?,
            epsilon,
        })
    };
    let biased = |tensors: &mut Tensors,
                  stem: &str,
                  outputs: u64,
                  inputs: u64,
                  transforms: Vec<ImportTransform>|
     -> Result<VisionLinear, FamilyError> {
        Ok(VisionLinear {
            weight: tensors.bind_transformed(
                &format!("{stem}.weight"),
                &[outputs, inputs],
                transforms.clone(),
            )?,
            bias: Some(tensors.bind_transformed(
                &format!("{stem}.bias"),
                &[outputs],
                transforms,
            )?),
            clamp: None,
        })
    };
    // Inkling 2's vision tower uses adjacent-pair 2-D axial RoPE (GGUF `mode=0`:
    // first half of each head rotates by column, second half by row); our vision
    // rotary kernel pairs column `i < W/2` with `i + W/2`, so deinterleave Q/K.
    let rotary = vec![rotary::half_split_rows(width, width)];

    let patch_frame = tensors.bind("v.patch_embd.weight", &[hidden, 3, patch, patch])?;
    let positions = VisionPositions {
        table: tensors.bind("v.position_embd.weight", &[max_patches, hidden])?,
        sampling: PositionSampling::PixelCenters { side },
    };
    let pre_norm = layer_norm(&mut tensors, "v.pre_ln")?;

    // Active CLS token & its position embedding (`cls = class_embd + class_position_embd`),
    // prepended at row 0 before `v.pre_ln`, attended by all ViT blocks at RoPE `(0, 0)`,
    // and sliced off before the spatial merger.
    let registers = bind_optional_class_vector(&mut tensors, "v.class_embd", hidden)?;
    let register_positions =
        bind_optional_class_vector(&mut tensors, "v.class_position_embd.weight", hidden)?;
    if register_positions.is_some() && registers.is_none() {
        return Err(invalid(
            "inkling2v v.class_position_embd.weight requires v.class_embd",
        ));
    }

    let stem = VisionStem::Patch {
        frames: vec![patch_frame],
        bias: None,
        positions: Some(positions),
        norm: Some(pre_norm),
        registers,
        register_positions,
    };

    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let ls1: Option<WeightDescriptor> = if tensors.contains(&format!("{p}ls1.weight")) {
            Some(tensors.bind(&format!("{p}ls1.weight"), &[hidden])?)
        } else {
            None
        };
        let ls2: Option<WeightDescriptor> = if tensors.contains(&format!("{p}ls2.weight")) {
            Some(tensors.bind(&format!("{p}ls2.weight"), &[hidden])?)
        } else {
            None
        };
        blocks.push(VisionBlock {
            operator_norm: layer_norm(&mut tensors, &format!("{p}ln1"))?,
            operator: VisionOperator::Attention(VisionAttention {
                heads,
                kv_heads: heads,
                width,
                query: biased(&mut tensors, &format!("{p}attn_q"), hidden, hidden, rotary.clone())?,
                key: biased(&mut tensors, &format!("{p}attn_k"), hidden, hidden, rotary.clone())?,
                value: biased(&mut tensors, &format!("{p}attn_v"), hidden, hidden, Vec::new())?,
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: Some(rotary_base),
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: biased(&mut tensors, &format!("{p}attn_out"), hidden, hidden, Vec::new())?,
            }),
            operator_post_norm: None,
            feedforward_norm: layer_norm(&mut tensors, &format!("{p}ln2"))?,
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(biased(
                    &mut tensors,
                    &format!("{p}ffn_up"),
                    intermediate,
                    hidden,
                    Vec::new(),
                )?),
                activation: VisionActivation::GeluErf,
                down: biased(&mut tensors, &format!("{p}ffn_down"), hidden, intermediate, Vec::new())?,
            },
            feedforward_post_norm: None,
            ls1,
            ls2,
        });
    }

    let merged_width = product(&[merge, merge, hidden])?;
    let stage1_out = tensors.rows("mm.1.weight")?;
    if let Some(proj_hidden) = meta.optional_integer("clip.vision.projector.hidden_size")? {
        if proj_hidden != stage1_out {
            return Err(invalid(format!(
                "inkling2v clip.vision.projector.hidden_size ({proj_hidden}) disagrees with mm.1.weight rows ({stage1_out})"
            )));
        }
    }
    let merger_bias = match meta.optional_bool("clip.vision.projector.use_bias")? {
        Some(flag) => flag,
        None => tensors.contains("mm.1.bias"),
    };
    let merger_linear = |tensors: &mut Tensors,
                         stem: &str,
                         outputs: u64,
                         inputs: u64|
     -> Result<VisionLinear, FamilyError> {
        Ok(VisionLinear {
            weight: tensors.bind(&format!("{stem}.weight"), &[outputs, inputs])?,
            bias: if merger_bias {
                Some(tensors.bind(&format!("{stem}.bias"), &[outputs])?)
            } else {
                None
            },
            clamp: None,
        })
    };
    let merger = VisionMerger {
        norm: None,
        reduction: CellReduction::Concatenate,
        standardize: None,
        projection_norm: None,
        stages: vec![
            MergerStage {
                linear: merger_linear(&mut tensors, "mm.1", stage1_out, merged_width)?,
                norm: None,
                activation: Some(VisionActivation::GeluErf),
            },
            MergerStage {
                linear: merger_linear(&mut tensors, "mm.2", output_hidden, stage1_out)?,
                norm: None,
                activation: None,
            },
        ],
        feed_forward: None,
        output_norm: None,
    };
    tensors.finish()?;

    let description = Box::new(VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PatchBudget { max_patches },
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

#[cfg(test)]
mod tests {
    use super::describe_projector;
    use magnitude_artifacts::gguf::{
        ByteOrder, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value,
    };

    fn kv_str(name: &str, value: &str) -> Metadata {
        Metadata {
            name: name.into(),
            value: Value::Scalar(Scalar::String(value.into())),
        }
    }

    fn kv_u32(name: &str, value: u64) -> Metadata {
        Metadata {
            name: name.into(),
            value: Value::Scalar(Scalar::Unsigned(value)),
        }
    }

    fn kv_f32(name: &str, value: f64) -> Metadata {
        Metadata {
            name: name.into(),
            value: Value::Scalar(Scalar::Float(value)),
        }
    }

    fn kv_bool(name: &str, value: bool) -> Metadata {
        Metadata {
            name: name.into(),
            value: Value::Scalar(Scalar::Bool(value)),
        }
    }

    fn tensor(name: impl Into<String>, shape: Vec<u64>, encoding: Encoding) -> TensorDescriptor {
        TensorDescriptor {
            name: name.into(),
            shape,
            encoding,
            offset: 0,
            nbytes: 0,
        }
    }

    fn projector_dir(
        class_embd_shape: Option<Vec<u64>>,
        class_pos_shape: Option<Vec<u64>>,
    ) -> Directory {
        let vh = 64u64;
        let vinter = 128u64;
        let heads = 4u64;
        let patch = 4u64;
        let merge = 2u64;
        let side = 4u64;
        let out_h = 128u64;

        let mut tensors = vec![
            tensor("v.patch_embd.weight", vec![vh, 3, patch, patch], Encoding::BF16),
            tensor("v.position_embd.weight", vec![side * side, vh], Encoding::BF16),
            tensor("v.pre_ln.weight", vec![vh], Encoding::F32),
            tensor("v.pre_ln.bias", vec![vh], Encoding::F32),
            tensor("v.blk.0.ln1.weight", vec![vh], Encoding::F32),
            tensor("v.blk.0.ln1.bias", vec![vh], Encoding::F32),
            tensor("v.blk.0.attn_q.weight", vec![vh, vh], Encoding::BF16),
            tensor("v.blk.0.attn_q.bias", vec![vh], Encoding::BF16),
            tensor("v.blk.0.attn_k.weight", vec![vh, vh], Encoding::BF16),
            tensor("v.blk.0.attn_k.bias", vec![vh], Encoding::BF16),
            tensor("v.blk.0.attn_v.weight", vec![vh, vh], Encoding::BF16),
            tensor("v.blk.0.attn_v.bias", vec![vh], Encoding::BF16),
            tensor("v.blk.0.attn_out.weight", vec![vh, vh], Encoding::BF16),
            tensor("v.blk.0.attn_out.bias", vec![vh], Encoding::BF16),
            tensor("v.blk.0.ln2.weight", vec![vh], Encoding::F32),
            tensor("v.blk.0.ln2.bias", vec![vh], Encoding::F32),
            tensor("v.blk.0.ffn_up.weight", vec![vinter, vh], Encoding::BF16),
            tensor("v.blk.0.ffn_up.bias", vec![vinter], Encoding::BF16),
            tensor("v.blk.0.ffn_down.weight", vec![vh, vinter], Encoding::BF16),
            tensor("v.blk.0.ffn_down.bias", vec![vh], Encoding::BF16),
            tensor("mm.1.weight", vec![vinter, merge * merge * vh], Encoding::BF16),
            tensor("mm.1.bias", vec![vinter], Encoding::BF16),
            tensor("mm.2.weight", vec![out_h, vinter], Encoding::BF16),
            tensor("mm.2.bias", vec![out_h], Encoding::BF16),
        ];
        if let Some(shape) = class_embd_shape {
            tensors.push(tensor("v.class_embd", shape, Encoding::BF16));
        }
        if let Some(shape) = class_pos_shape {
            tensors.push(tensor("v.class_position_embd.weight", shape, Encoding::BF16));
        }

        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: vec![
                kv_str("general.architecture", "clip"),
                kv_str("general.type", "mmproj"),
                kv_bool("clip.has_vision_encoder", true),
                kv_str("clip.projector_type", "inkling2v"),
                kv_u32("clip.vision.projection_dim", out_h),
                kv_u32("clip.vision.image_size", side * patch),
                kv_u32("clip.vision.patch_size", patch),
                kv_u32("clip.vision.embedding_length", vh),
                kv_u32("clip.vision.feed_forward_length", vinter),
                kv_u32("clip.vision.block_count", 1),
                kv_u32("clip.vision.attention.head_count", heads),
                Metadata {
                    name: "clip.vision.image_mean".into(),
                    value: Value::Array(vec![0.5f64, 0.5, 0.5].into()),
                },
                Metadata {
                    name: "clip.vision.image_std".into(),
                    value: Value::Array(vec![0.5f64, 0.5, 0.5].into()),
                },
                kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6),
                kv_u32("clip.vision.spatial_merge_size", merge),
            ],
            tensors,
        }
    }

    #[test]
    fn class_embedding_and_position_1d_and_2d_are_consistent() {
        let none = describe_projector(&projector_dir(None, None), 128).expect("absent class tensors");
        let flat = describe_projector(&projector_dir(Some(vec![64]), Some(vec![64])), 128)
            .expect("1-D class tensors");
        let batched = describe_projector(&projector_dir(Some(vec![1, 64]), Some(vec![1, 64])), 128)
            .expect("2-D [1, hidden] class tensors");

        assert_eq!(none.stem.registers(), 0);
        assert_eq!(flat.stem.registers(), 1);
        assert_eq!(batched.stem.registers(), 1);
        assert_eq!(
            flat.stem.registers(),
            batched.stem.registers(),
            "1-D and 2-D [1, hidden] class tensors both register 1 CLS row"
        );
    }

    #[test]
    fn class_embedding_and_position_reject_invalid_shapes() {
        for bad in [vec![2, 64], vec![0, 64], vec![64, 1], vec![65]] {
            let err = describe_projector(&projector_dir(Some(bad.clone()), None), 128)
                .expect_err("invalid v.class_embd shape must be rejected");
            assert!(
                err.to_string().contains("v.class_embd"),
                "shape {bad:?}: {err}"
            );

            let err = describe_projector(&projector_dir(None, Some(bad.clone())), 128)
                .expect_err("invalid v.class_position_embd.weight shape must be rejected");
            assert!(
                err.to_string().contains("v.class_position_embd.weight"),
                "shape {bad:?}: {err}"
            );
        }
    }
}
