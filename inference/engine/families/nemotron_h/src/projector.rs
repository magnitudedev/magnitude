//! The Nemotron Omni projector (`nemotron_omni`): a C-RADIO ViT tower and a
//! squared-ReLU MLP over 2 × 2 pixel-shuffled cells.
//!
//! The tower embeds each 16-pixel patch without bias, adds the learned
//! 128 × 128 position grid resampled to the image (cropped position
//! embedding), and places its learned class and register rows ahead of the
//! patches. Its blocks are pre-norm (layer norms with bias) with biased
//! projections, full attention without rotary and an erf-GELU feed-forward.
//! The merger drops the class and register rows, optionally layer-normalizes
//! each patch row (the released `vision_final_layernorm`, present when the
//! language model was trained with MTP), concatenates each 2 × 2 cell in
//! raster order (the released `pixel_shuffle`, version 2), RMS-normalizes the
//! cell and projects it through `Linear → ReLU² → Linear`, all without bias.
//!
//! The stored video patch kernel (`v.patch_embd.weight.1`, a temporal pair of
//! frames) is shape-checked and given no role: still images use the image
//! patch kernel only.

use crate::Error;
use magnitude_artifacts::gguf::{Directory, Element, Scalar, Value};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    checked_product, ActivationDType, CellReduction, ImportTransform, MergerStage,
    PositionSampling, RowRange, VisionActivation, VisionAttention, VisionAttentionScale,
    VisionAttentionSpan, VisionBlock, VisionDescription, VisionFeedForward, VisionLinear,
    VisionMerger, VisionNorm, VisionPositions, VisionPreprocessing, VisionResampling, VisionResize,
    VisionStem, VisionUp, WeightDescriptor,
};

/// Every projector metadata key the family reads, beside `general.*`.
const KEYS: &[&str] = &[
    "clip.has_vision_encoder",
    "clip.projector_type",
    "clip.use_gelu",
    "clip.vision.nemotron_omni",
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
    "clip.vision.projector.scale_factor",
    "clip.vision.image_min_pixels",
    "clip.vision.image_max_pixels",
];

/// The released projector MLP's RMS norm epsilon (`NemotronH_Omni_RMSNorm`
/// of `mlp1`); the converter does not store it.
const PROJECTION_NORM_EPSILON: f64 = 1e-5;

fn invalid(message: impl Into<String>) -> Error {
    Error::Projector(message.into())
}

struct Reader<'a> {
    directory: &'a Directory,
    tensors: Tensors<'a>,
    epsilon: f64,
}

impl Reader<'_> {
    fn count(&self, key: &str) -> Result<u64, Error> {
        self.directory
            .value(key)
            .and_then(Value::unsigned)
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid(format!("{key} must be a positive integer")))
    }

    fn flag(&self, key: &str) -> Option<bool> {
        match self.directory.value(key) {
            Some(Value::Scalar(Scalar::Bool(value))) => Some(*value),
            _ => None,
        }
    }

    fn triple(&self, key: &str) -> Result<[f64; 3], Error> {
        let Some(Value::Array(values)) = self.directory.value(key) else {
            return Err(invalid(format!("{key} must contain three numbers")));
        };
        values
            .iter()
            .map(|value| match value {
                Element::Float(value) if value.is_finite() => Some(value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.try_into().ok())
            .ok_or_else(|| invalid(format!("{key} must contain three finite numbers")))
    }

    /// Image pixel bound `key` as whole patches.
    fn patches(&self, key: &str, patch: u64) -> Result<u64, Error> {
        let pixels = self.count(key)?;
        let area = patch * patch;
        if !pixels.is_multiple_of(area) {
            return Err(invalid(format!("{key} is not a whole number of patches")));
        }
        Ok(pixels / area)
    }

    fn layer_norm(&mut self, name: &str, width: u64) -> Result<VisionNorm, Error> {
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

    fn unbiased(&mut self, name: &str, outputs: u64, inputs: u64) -> Result<VisionLinear, Error> {
        Ok(VisionLinear {
            weight: self.tensors.bind(name, &[outputs, inputs])?,
            bias: None,
            clamp: None,
        })
    }

    /// The query, key and value of block `p`: stored apart, or as one fused
    /// projection imported as three row ranges.
    fn query_key_value(&mut self, p: &str, hidden: u64) -> Result<[VisionLinear; 3], Error> {
        let fused = format!("{p}.attn_qkv");
        if !self.tensors.contains(&format!("{fused}.weight")) {
            return Ok([
                self.linear(&format!("{p}.attn_q"), hidden, hidden)?,
                self.linear(&format!("{p}.attn_k"), hidden, hidden)?,
                self.linear(&format!("{p}.attn_v"), hidden, hidden)?,
            ]);
        }
        let rows = checked_product(&[3, hidden])?;
        let weight = self
            .tensors
            .bind(&format!("{fused}.weight"), &[rows, hidden])?;
        let bias = self.tensors.bind(&format!("{fused}.bias"), &[rows])?;
        let part = |stored: &WeightDescriptor, index: u64| {
            let mut shape = stored.shape.clone();
            shape[0] = hidden;
            WeightDescriptor {
                name: stored.name.clone(),
                shape,
                transforms: vec![ImportTransform::Rows(RowRange {
                    start: index * hidden,
                    rows: hidden,
                })],
            }
        };
        Ok([0, 1, 2].map(|index| VisionLinear {
            weight: part(&weight, index),
            bias: Some(part(&bias, index)),
            clamp: None,
        }))
    }
}

/// Interpret a Nemotron Omni projector inventory without importing it.
pub fn describe_projector(directory: &Directory) -> Result<VisionDescription, Error> {
    if directory
        .value("general.architecture")
        .and_then(Value::string)
        != Some("clip")
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory
            .value("clip.projector_type")
            .and_then(Value::string)
            != Some("nemotron_omni")
        || directory.value("clip.has_vision_encoder") != Some(&Value::Scalar(Scalar::Bool(true)))
    {
        return Err(invalid("unsupported Nemotron projector identity"));
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
    let epsilon = match directory.value("clip.vision.attention.layer_norm_epsilon") {
        Some(Value::Scalar(Scalar::Float(value))) if value.is_finite() && *value > 0.0 => *value,
        _ => return Err(invalid("projector declares no positive layer norm epsilon")),
    };
    let mut r = Reader {
        directory,
        tensors: Tensors::new(directory),
        epsilon,
    };
    // The released tower's feed-forward is the erf GELU.
    if r.flag("clip.use_gelu") != Some(true) {
        return Err(invalid("the Nemotron tower requires its GELU feed-forward"));
    }
    if directory.value("clip.vision.nemotron_omni").is_some()
        && r.flag("clip.vision.nemotron_omni") != Some(true)
    {
        return Err(invalid("clip.vision.nemotron_omni must be true"));
    }
    let hidden = r.count("clip.vision.embedding_length")?;
    let output_hidden = r.count("clip.vision.projection_dim")?;
    let intermediate = r.count("clip.vision.feed_forward_length")?;
    let depth = r.count("clip.vision.block_count")?;
    let heads = r.count("clip.vision.attention.head_count")?;
    let patch = r.count("clip.vision.patch_size")?;
    let merge = r.count("clip.vision.projector.scale_factor")?;
    r.count("clip.vision.image_size")?;
    if !hidden.is_multiple_of(heads)
        || !(hidden / heads).is_multiple_of(4)
        || depth > directory.tensors.len() as u64
    {
        return Err(invalid("invalid Nemotron tower geometry"));
    }
    let min_patches = r.patches("clip.vision.image_min_pixels", patch)?;
    let max_patches = r.patches("clip.vision.image_max_pixels", patch)?;

    let frame = r
        .tensors
        .bind("v.patch_embd.weight", &[hidden, 3, patch, patch])?;
    // The video kernel convolves a temporal pair of frames; validated only.
    if r.tensors.contains("v.patch_embd.weight.1") {
        r.tensors
            .bind("v.patch_embd.weight.1", &[hidden, 6, patch, patch])?;
    }
    let (table, stored) = match r.tensors.shape("v.position_embd.weight")? {
        [rows, width] => (*rows, vec![*rows, *width]),
        [1, rows, width] => (*rows, vec![1, *rows, *width]),
        _ => return Err(invalid("position embedding must be a matrix")),
    };
    let side = (table as f64).sqrt() as u64;
    if side.checked_mul(side) != Some(table) || stored.last() != Some(&hidden) {
        return Err(invalid(
            "position embedding must be a square grid of tower rows",
        ));
    }
    let transforms = if stored.len() == 3 {
        vec![ImportTransform::DropUnitAxis]
    } else {
        Vec::new()
    };
    let positions = VisionPositions {
        table: r
            .tensors
            .bind_transformed("v.position_embd.weight", &stored, transforms)?,
        sampling: PositionSampling::CroppedSquare { side },
    };
    let registers = match r.tensors.shape("v.class_embd")? {
        [rows, width] if *rows > 0 && *width == hidden => *rows,
        _ => return Err(invalid("class and register rows must be tower rows")),
    };
    let stem = VisionStem::Patch {
        frames: vec![frame],
        bias: None,
        positions,
        norm: None,
        registers: Some(r.tensors.bind("v.class_embd", &[registers, hidden])?),
    };

    let mut blocks = Vec::with_capacity(depth as usize);
    for layer in 0..depth {
        let p = format!("v.blk.{layer}");
        let [query, key, value] = r.query_key_value(&p, hidden)?;
        blocks.push(VisionBlock {
            attention_norm: r.layer_norm(&format!("{p}.ln1"), hidden)?,
            attention: VisionAttention {
                heads,
                width: hidden / heads,
                query,
                key,
                value,
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: None,
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: r.linear(&format!("{p}.attn_out"), hidden, hidden)?,
            },
            attention_post_norm: None,
            feedforward_norm: r.layer_norm(&format!("{p}.ln2"), hidden)?,
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(r.linear(&format!("{p}.ffn_up"), intermediate, hidden)?),
                activation: VisionActivation::GeluErf,
                down: r.linear(&format!("{p}.ffn_down"), hidden, intermediate)?,
            },
            feedforward_post_norm: None,
        });
    }

    let cell = checked_product(&[merge, merge, hidden])?;
    let projection = r.tensors.rows("mm.model.mlp.1.weight")?;
    let merger = VisionMerger {
        norm: if r.tensors.contains("v.post_ln.weight") {
            Some(r.layer_norm("v.post_ln", hidden)?)
        } else {
            None
        },
        reduction: CellReduction::Concatenate,
        standardize: None,
        projection_norm: Some(VisionNorm::Rms {
            weight: Some(r.tensors.bind("mm.model.mlp.0.weight", &[cell])?),
            epsilon: PROJECTION_NORM_EPSILON,
        }),
        stages: vec![
            MergerStage {
                linear: r.unbiased("mm.model.mlp.1.weight", projection, cell)?,
                activation: Some(VisionActivation::ReluSquared),
            },
            MergerStage {
                linear: r.unbiased("mm.model.mlp.3.weight", output_hidden, projection)?,
                activation: None,
            },
        ],
        output_norm: None,
    };
    let preprocessing = VisionPreprocessing {
        resize: VisionResize::PatchBounds {
            min_patches,
            max_patches,
        },
        // The released processor resamples with PyTorch's antialiased
        // bicubic (a = −0.5, scale-aware support), the same filter; it keeps
        // the samples unquantized where this one rounds them to 8 bits, at
        // most half an 8-bit step per channel.
        resampling: VisionResampling::Bicubic,
        mean: r.triple("clip.vision.image_mean")?,
        std: r.triple("clip.vision.image_std")?,
        channels: 3,
        patch,
        merge,
    };
    r.tensors.finish()?;
    let description = VisionDescription {
        // The squared ReLU publishes the square of its input, which f16
        // (at most 65504) holds only below 256; bf16 intermediates around
        // the F32 residual stream keep the released bf16 forward's range.
        activation_dtype: ActivationDType::BF16,
        hidden,
        output_hidden,
        preprocessing,
        stem,
        window: None,
        blocks,
        merger,
    };
    description.validate()?;
    Ok(description)
}
