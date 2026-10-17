//! MiniCPM-V 4.7's 4× SigLIP projector and canvas input interpretation.
pub mod inputs;

use crate::{Error, invalid};
use magnitude_artifacts::gguf::{Directory, PROJECTOR_ARCHITECTURE, Scalar, Value};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    ActivationDType, CellReduction, MergerStage, PositionSampling, VisionActivation,
    VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock, VisionDescription,
    VisionFeedForward, VisionLinear, VisionMerger, VisionNorm, VisionPositions,
    VisionPreprocessing, VisionResampling, VisionResize, VisionStem, VisionUp, checked_product,
};

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
    "clip.vision.image_mean",
    "clip.vision.image_std",
    "clip.vision.projector.scale_factor",
    "clip.vision.max_slice_nums",
    "clip.vision.wa_layer_indexes",
    "clip.use_gelu",
    "clip.vision.attention.layer_norm_epsilon",
];

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
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid(format!("{key} must be a positive integer")))
    }
    fn triple(&self, key: &str) -> Result<[f64; 3], Error> {
        let Some(Value::Array(values)) = self.directory.value(key) else {
            return Err(invalid(format!("{key} must contain three numbers")));
        };
        let values = values
            .iter()
            .map(|v| match v {
                Scalar::Float(x) if x.is_finite() => Some(*x),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| invalid(format!("{key} must contain three finite numbers")))?;
        Ok(values)
    }
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

pub fn describe_projector(directory: &Directory) -> Result<VisionDescription, Error> {
    if directory
        .value("general.architecture")
        .and_then(Value::string)
        != Some(PROJECTOR_ARCHITECTURE)
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory
            .value("clip.projector_type")
            .and_then(Value::string)
            != Some("minicpmv4_7")
        || directory.value("clip.has_vision_encoder") != Some(&Value::Scalar(Scalar::Bool(true)))
    {
        return Err(invalid("unsupported MiniCPM-V projector identity"));
    }
    if let Some(entry) = directory
        .metadata
        .iter()
        .find(|e| !e.name.starts_with("general.") && !KEYS.contains(&e.name.as_str()))
    {
        return Err(invalid(format!(
            "unknown projector metadata {:?}",
            entry.name
        )));
    }
    let epsilon = match directory.value("clip.vision.attention.layer_norm_epsilon") {
        Some(Value::Scalar(Scalar::Float(v))) if v.is_finite() && *v > 0.0 => *v,
        _ => return Err(invalid("projector declares no positive layer norm epsilon")),
    };
    let mut r = Reader {
        directory,
        tensors: Tensors::new(directory),
        epsilon,
    };
    if r.count("clip.vision.projector.scale_factor")? != 2 {
        return Err(invalid(
            "MiniCPM-V 16x intra-tower compression is not supported by this projector",
        ));
    }
    if directory.value("clip.use_gelu") != Some(&Value::Scalar(Scalar::Bool(true))) {
        return Err(invalid("MiniCPM-V requires the published GELU tower"));
    }
    let hidden = r.count("clip.vision.embedding_length")?;
    let output_hidden = r.count("clip.vision.projection_dim")?;
    let intermediate = r.count("clip.vision.feed_forward_length")?;
    let depth = r.count("clip.vision.block_count")?;
    let heads = r.count("clip.vision.attention.head_count")?;
    let patch = r.count("clip.vision.patch_size")?;
    if hidden % heads != 0 || (hidden / heads) % 4 != 0 || depth > directory.tensors.len() as u64 {
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
        positions,
        norm: None,
    };
    let mut blocks = Vec::new();
    for layer in 0..depth {
        let p = format!("v.blk.{layer}");
        blocks.push(VisionBlock {
            attention_norm: r.norm(&format!("{p}.ln1"), hidden)?,
            attention: VisionAttention {
                heads,
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
            },
            attention_post_norm: None,
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
                linear: r.linear("mm.up", joined, joined)?,
                activation: Some(VisionActivation::GeluErf),
            },
            MergerStage {
                linear: r.linear("mm.down", output_hidden, joined)?,
                activation: None,
            },
        ],
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
            resolution: r.count("clip.vision.image_size")?,
            max_slices: r.count("clip.vision.max_slice_nums")?,
        },
        resampling: VisionResampling::Bicubic,
        mean: r.triple("clip.vision.image_mean")?,
        std: r.triple("clip.vision.image_std")?,
        channels: 3,
        patch,
        merge: 2,
    };
    r.tensors.finish().map_err(Error::from)?;
    let description = VisionDescription {
        activation_dtype: ActivationDType::F16,
        hidden,
        output_hidden,
        preprocessing,
        stem,
        window: None,
        blocks,
        merger,
    };
    description.validate().map_err(|e| invalid(e.to_string()))?;
    Ok(description)
}
