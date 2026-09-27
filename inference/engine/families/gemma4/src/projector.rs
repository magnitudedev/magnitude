//! Gemma 4 projectors: `gemma4v` (a ViT with RMS sandwich blocks, E2B/E4B
//! and 26B/31B) and `gemma4uv` (the encoder-free patch projection of the
//! unified 12B). Audio components that share the projector file are not
//! interpreted; every vision tensor is bound once.

use crate::{invalid, product, Error};
use magnitude_artifacts::gguf::{Directory, Scalar, Value};
use magnitude_family_common::{HeaderError, Tensors};
use magnitude_family_contracts::{
    CellReduction, ImportTransform, LinearClamp, MergerStage, PositionSampling, RmsNorm,
    Standardize, UnweightedRms, VisionActivation, VisionAttention, VisionAttentionScale,
    VisionAttentionSpan, VisionBlock, VisionDescription, VisionFeedForward, VisionLinear,
    VisionMerger, VisionNorm, VisionPositions, VisionPreprocessing, VisionResampling, VisionResize,
    VisionStem, VisionUp, WeightDescriptor, ActivationDType,
};

/// The released vision configuration's facts the header does not record:
/// the pooling kernel (`pooling_kernel_size`), the rotary base
/// (`rope_parameters.rope_theta`), the soft-token budget of the image
/// processor (`max_soft_tokens`) and the feed-forward activation
/// (`hidden_activation = gelu_pytorch_tanh`; llama.cpp resolves GELU_QUICK
/// because the header has no `clip.use_gelu`, and departs from the model).
const POOLING: u64 = 3;
const ROTARY_BASE: f64 = 100.0;
const SOFT_TOKENS: u64 = 280;
const ACTIVATION: VisionActivation = VisionActivation::GeluTanh;

/// The unified embedder's layer norms are PyTorch's default (eps 1e-5); the
/// header's epsilon is the RMS norms'.
const LAYER_NORM_EPSILON: f64 = 1e-5;

/// Every `clip.` metadata key a vision interpretation reads or may ignore
/// (the audio encoder's keys describe a component that is not interpreted).
const IGNORED_PREFIXES: &[&str] = &["clip.audio.", "clip.has_audio_encoder"];

struct Reader<'a> {
    directory: &'a Directory,
    tensors: Tensors<'a>,
}

impl Reader<'_> {
    fn unsigned(&self, key: &str) -> Result<u64, Error> {
        self.directory
            .value(key)
            .and_then(Value::unsigned)
            .ok_or_else(|| invalid(format!("projector metadata {key} must be an integer")))
    }

    fn number(&self, key: &str) -> Result<f64, Error> {
        match self.directory.value(key) {
            Some(Value::Scalar(Scalar::Float(value))) if value.is_finite() => Ok(*value),
            _ => Err(invalid(format!("projector metadata {key} must be a number"))),
        }
    }

    fn triple(&self, key: &str) -> Result<[f64; 3], Error> {
        let values = match self.directory.value(key) {
            Some(Value::Array(values)) if values.len() == 3 => values
                .iter()
                .map(|value| match value {
                    Scalar::Float(value) if value.is_finite() => Some(*value),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        }
        .ok_or_else(|| invalid(format!("projector metadata {key} must hold three numbers")))?;
        Ok([values[0], values[1], values[2]])
    }

    fn tensor(&mut self, name: &str, shape: &[u64]) -> Result<WeightDescriptor, Error> {
        Ok(self.tensors.bind(name, shape)?)
    }

    fn present(&self, name: &str) -> bool {
        self.tensors.contains(name)
    }

    /// A bias-free linear `[outputs, inputs]`, clippable when its four
    /// bounds are stored.
    fn linear(&mut self, stem: &str, outputs: u64, inputs: u64) -> Result<VisionLinear, Error> {
        let weight = self.tensor(&format!("{stem}.weight"), &[outputs, inputs])?;
        let clamp = if self.present(&format!("{stem}.input_min")) {
            Some(LinearClamp {
                input_minimum: self.tensor(&format!("{stem}.input_min"), &[1])?,
                input_maximum: self.tensor(&format!("{stem}.input_max"), &[1])?,
                output_minimum: self.tensor(&format!("{stem}.output_min"), &[1])?,
                output_maximum: self.tensor(&format!("{stem}.output_max"), &[1])?,
            })
        } else {
            None
        };
        Ok(VisionLinear {
            weight,
            bias: None,
            clamp,
        })
    }

    /// Every vision tensor (`v.`, `mm.` except the audio embedder's
    /// `mm.a.`) is bound.
    fn finish(self) -> Result<(), Error> {
        let vision =
            |name: &&str| name.starts_with("v.") || (name.starts_with("mm.") && !name.starts_with("mm.a."));
        if let Some(name) = self.tensors.unbound().find(vision) {
            return Err(HeaderError::UnboundWeight(name.to_owned()).into());
        }
        Ok(())
    }
}

/// The two-axis rotary head rows of Gemma 4 (HF's axial rope: columns
/// `[0, 2P)` rotate by the patch column, `[2P, 4P)` by the patch row, each
/// half as `rotate_half` pairs `(f, f + P)`) in the vision contract's
/// layout (column `i < 2P` pairs with `i + 2P`, axis `i / P`): logical row
/// `a · P + f` is stored row `a · 2P + f`, and `2P + a · P + f` is
/// `a · 2P + P + f`.
fn axial_order(width: u64) -> Vec<u64> {
    let quarter = width / 4;
    let half = (0..2)
        .flat_map(|axis| (0..quarter).map(move |frequency| axis * 2 * quarter + frequency));
    half.clone()
        .chain(half.map(|stored| stored + quarter))
        .collect()
}

fn permuted(mut weight: WeightDescriptor, order: &[u64]) -> WeightDescriptor {
    weight.transforms.push(ImportTransform::PermuteRows {
        order: order.to_vec(),
    });
    weight
}

/// Interpret a Gemma 4 projector.
pub fn describe_projector(
    directory: &Directory,
    output_hidden: u64,
) -> Result<VisionDescription, Error> {
    if directory.value("general.architecture").and_then(Value::string) != Some("clip")
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory.value("clip.has_vision_encoder")
            != Some(&Value::Scalar(Scalar::Bool(true)))
    {
        return Err(invalid("unsupported Gemma projector identity"));
    }
    for entry in &directory.metadata {
        let known = entry.name.starts_with("general.")
            || IGNORED_PREFIXES
                .iter()
                .any(|prefix| entry.name.starts_with(prefix))
            || [
                "clip.has_vision_encoder",
                "clip.vision.projector_type",
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
            ]
            .contains(&entry.name.as_str());
        if !known {
            return Err(invalid(format!(
                "unknown projector metadata key {:?}",
                entry.name
            )));
        }
    }
    let mut reader = Reader {
        directory,
        tensors: Tensors::new(directory),
    };
    if reader.unsigned("clip.vision.projection_dim")? != output_hidden {
        return Err(invalid("projector width differs from the decoder width"));
    }
    let kind = directory
        .value("clip.vision.projector_type")
        .and_then(Value::string)
        .ok_or_else(|| invalid("Gemma projector declares no projector type"))?;
    let description = match kind {
        "gemma4v" => encoder(&mut reader, output_hidden)?,
        "gemma4uv" => unified(&mut reader, output_hidden)?,
        kind => return Err(invalid(format!("unsupported Gemma projector type {kind:?}"))),
    };
    reader.finish()?;
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}

/// `gemma4v`: patch convolution plus per-axis position tables, RMS sandwich
/// blocks, 3×3 average pooling × √C, an optional standardization, an
/// unweighted RMS and one linear.
fn encoder(reader: &mut Reader, output_hidden: u64) -> Result<VisionDescription, Error> {
    let hidden = reader.unsigned("clip.vision.embedding_length")?;
    let intermediate = reader.unsigned("clip.vision.feed_forward_length")?;
    let depth = reader.unsigned("clip.vision.block_count")?;
    let heads = reader.unsigned("clip.vision.attention.head_count")?;
    let patch = reader.unsigned("clip.vision.patch_size")?;
    let epsilon = reader.number("clip.vision.attention.layer_norm_epsilon")?;
    if heads == 0 || hidden % heads != 0 || (hidden / heads) % 4 != 0 || patch == 0 {
        return Err(invalid("projector heads do not split into 2-D rotary heads"));
    }
    let width = hidden / heads;
    let order = axial_order(width);
    let table = directory_table(reader, hidden)?;
    let rms = |weight: WeightDescriptor| VisionNorm::Rms {
        weight: Some(weight),
        epsilon,
    };
    let mut blocks = Vec::new();
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let query = reader.linear(&format!("{p}attn_q"), hidden, hidden)?;
        let key = reader.linear(&format!("{p}attn_k"), hidden, hidden)?;
        blocks.push(VisionBlock {
            attention_norm: rms(reader.tensor(&format!("{p}ln1.weight"), &[hidden])?),
            attention: VisionAttention {
                heads,
                width,
                query: VisionLinear {
                    weight: permuted(query.weight, &order),
                    ..query
                },
                key: VisionLinear {
                    weight: permuted(key.weight, &order),
                    ..key
                },
                value: reader.linear(&format!("{p}attn_v"), hidden, hidden)?,
                query_norm: Some(RmsNorm {
                    weight: permuted(
                        reader.tensor(&format!("{p}attn_q_norm.weight"), &[width])?,
                        &order,
                    ),
                    epsilon,
                }),
                key_norm: Some(RmsNorm {
                    weight: permuted(
                        reader.tensor(&format!("{p}attn_k_norm.weight"), &[width])?,
                        &order,
                    ),
                    epsilon,
                }),
                value_norm: Some(UnweightedRms { epsilon }),
                rotary_base: ROTARY_BASE,
                scale: VisionAttentionScale::Unit,
                span: VisionAttentionSpan::Full,
                output: reader.linear(&format!("{p}attn_out"), hidden, hidden)?,
            },
            attention_post_norm: Some(rms(
                reader.tensor(&format!("{p}attn_post_norm.weight"), &[hidden])?
            )),
            feedforward_norm: rms(reader.tensor(&format!("{p}ln2.weight"), &[hidden])?),
            feedforward: VisionFeedForward {
                up: VisionUp::Gated {
                    gate: reader.linear(&format!("{p}ffn_gate"), intermediate, hidden)?,
                    up: reader.linear(&format!("{p}ffn_up"), intermediate, hidden)?,
                },
                activation: ACTIVATION,
                down: reader.linear(&format!("{p}ffn_down"), hidden, intermediate)?,
            },
            feedforward_post_norm: Some(rms(
                reader.tensor(&format!("{p}ffn_post_norm.weight"), &[hidden])?
            )),
        });
    }
    let standardize = if reader.present("v.std_bias") {
        Some(Standardize {
            bias: reader.tensor("v.std_bias", &[hidden])?,
            scale: reader.tensor("v.std_scale", &[hidden])?,
        })
    } else {
        None
    };
    Ok(VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PatchBudget {
                max_patches: SOFT_TOKENS * POOLING * POOLING,
            },
            resampling: VisionResampling::Bicubic,
            // The header records the processor's identity normalization;
            // the model's embedder maps pixels to 2 (p − ½).
            mean: [0.5; 3],
            std: [0.5; 3],
            channels: 3,
            patch,
            merge: POOLING,
        },
        stem: VisionStem::Patch {
            frames: vec![reader.tensor("v.patch_embd.weight", &[hidden, 3, patch, patch])?],
            bias: None,
            positions: table,
            norm: None,
        },
        window: None,
        blocks,
        merger: VisionMerger {
            norm: None,
            reduction: CellReduction::Average {
                scale: (hidden as f64).sqrt(),
            },
            standardize,
            projection_norm: Some(VisionNorm::Rms {
                weight: None,
                epsilon,
            }),
            stages: vec![MergerStage {
                linear: reader.linear("mm.input_projection", output_hidden, hidden)?,
                activation: None,
            }],
            output_norm: None,
        },
    })
}

/// `gemma4uv`: layer norm of the flattened 48-pixel patch (channel-major, as
/// the converter stores its columns), a linear with bias, a layer norm, the
/// per-axis position tables, a layer norm, an unweighted RMS and one linear.
fn unified(reader: &mut Reader, output_hidden: u64) -> Result<VisionDescription, Error> {
    let hidden = reader.unsigned("clip.vision.embedding_length")?;
    let epsilon = reader.number("clip.vision.attention.layer_norm_epsilon")?;
    if reader.unsigned("clip.vision.block_count")? != 0 {
        return Err(invalid("a unified Gemma projector has no encoder blocks"));
    }
    let patch = reader.unsigned("clip.vision.patch_size")? * POOLING;
    let pixels = product(&[3, patch, patch])?;
    let layer_norm = |reader: &mut Reader, stem: &str, width| -> Result<VisionNorm, Error> {
        Ok(VisionNorm::Layer {
            weight: reader.tensor(&format!("{stem}.weight"), &[width])?,
            bias: reader.tensor(&format!("{stem}.bias"), &[width])?,
            epsilon: LAYER_NORM_EPSILON,
        })
    };
    let input_norm = layer_norm(reader, "v.patch_norm.1", pixels)?;
    let projection = VisionLinear {
        weight: reader.tensor("v.patch_embd.weight", &[hidden, pixels])?,
        bias: Some(reader.tensor("v.patch_embd.bias", &[hidden])?),
        clamp: None,
    };
    let projection_norm = layer_norm(reader, "v.patch_norm.2", hidden)?;
    let positions = directory_table(reader, hidden)?;
    let position_norm = layer_norm(reader, "v.patch_norm.3", hidden)?;
    Ok(VisionDescription {
        activation_dtype: ActivationDType::BF16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PatchBudget {
                max_patches: SOFT_TOKENS,
            },
            resampling: VisionResampling::Bicubic,
            mean: reader.triple("clip.vision.image_mean")?,
            std: reader.triple("clip.vision.image_std")?,
            channels: 3,
            patch,
            merge: 1,
        },
        stem: VisionStem::NormalizedPatch {
            input_norm,
            projection,
            projection_norm,
            positions,
            position_norm,
        },
        window: None,
        blocks: Vec::new(),
        merger: VisionMerger {
            norm: None,
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: Some(VisionNorm::Rms {
                weight: None,
                epsilon,
            }),
            stages: vec![MergerStage {
                linear: reader.linear("mm.input_projection", output_hidden, hidden)?,
                activation: None,
            }],
            output_norm: None,
        },
    })
}

/// The per-axis position tables `[2, length, H]`.
fn directory_table(reader: &mut Reader, hidden: u64) -> Result<VisionPositions, Error> {
    let name = "v.position_embd.weight";
    let length = match reader.directory.tensor(name).map(|tensor| tensor.shape.as_slice()) {
        Some(&[2, length, width]) if width == hidden && length > 0 => length,
        _ => return Err(invalid("invalid projector position tables")),
    };
    Ok(VisionPositions {
        table: reader.tensor(name, &[2, length, hidden])?,
        sampling: PositionSampling::Axes { length },
    })
}
