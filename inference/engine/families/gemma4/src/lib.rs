//! Gemma 4 artifact recognition and family-specific numerical interpretation.
//!
//! This adapter recognizes `gemma4` GGUF metadata and binds every stored
//! tensor to the family-neutral decoder contract. Everything that varies
//! between Gemma 4 models is read from metadata and tensor shapes: the
//! window/full layer pattern, per-layer head widths and key/value heads,
//! per-layer feed-forward widths, layers that reuse an earlier layer's
//! history, layers whose values are their keys, per-layer inputs and routed
//! experts.

mod family;
pub mod inputs;
mod projector;

pub use family::Gemma4Family;
pub use projector::describe_projector;

use magnitude_artifacts::{
    gguf::{Directory, Value},
    Package, PackageIdentity,
};
use magnitude_family_common::{HeaderError, Metadata, Tensors};
use magnitude_family_contracts::{
    checked_product, ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Branch,
    BranchOutput, Decoder, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm,
    ExpertSelection, FamilyId, FeedForwardUp, HeadNorm, HistoryDomain, HistoryReads,
    ImportTransform, InputNorm, InputSemantics, KeyValue, MediaRowAttention, ModelDefinition,
    Operator, OutputForm, PerLayerEntry, PerLayerInput, ResidualForm, RmsNorm, Rotary,
    RotaryDivisors, RotaryPair, RouteNormalization, RoutedFfn, Router, RouterInput, RowRange,
    ScoreFunction, Sublayer, SublayerIndex,
    TextCoordinateSemantics, UnweightedRms, ValueNorm, ValueSource, WeightDescriptor,
};
use std::{error, fmt};

/// The GGUF architecture and family identity.
pub const ARCHITECTURE: &str = "gemma4";

/// Every `gemma4.` metadata key this family interprets. A key outside this
/// set would change the model in a way the definition does not express.
const METADATA_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "embedding_length_per_layer_input",
    "feed_forward_length",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_swa",
    "attention.value_length_swa",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.shared_kv_layers",
    "rope.freq_base",
    "rope.freq_base_swa",
    "rope.dimension_count",
    "rope.dimension_count_swa",
    "final_logit_softcapping",
];

/// Full-attention layers rotate only the first quarter of each head's pairs
/// ("proportional" rotary, `partial_rotary_factor = 0.25` in the released
/// configuration), with frequencies taken over the whole head width; the other
/// pairs pass through unrotated. The GGUF records this fraction only in the
/// values of `rope_freqs.weight` (divisor 1 for rotated pairs, 1e30 for the
/// rest), which headers do not carry; the importer checks those values
/// against the table at load.
const FULL_ROTATED_PAIR_FRACTION: u64 = 4;

/// Per-layer inputs of media rows read this table row in place of a token
/// (the released model maps every non-text id to row 0).
const PER_LAYER_MEDIA_ROW: u64 = 0;

/// Failure to recognize or interpret a Gemma 4 artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for Error {}

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        invalid(match error {
            HeaderError::UnknownMetadata(key) => {
                format!("Gemma artifact carries uninterpreted metadata {key}")
            }
            HeaderError::MissingWeight(name) => format!("missing Gemma weight {name:?}"),
            HeaderError::WeightShape {
                name,
                expected,
                received,
            } => format!("Gemma weight {name:?}: expected {expected:?}, received {received:?}"),
            HeaderError::UnboundWeight(name) => {
                format!("Gemma artifact contains unbound weight role {name:?}")
            }
            error => error.to_string(),
        })
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}

fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

/// Recognize a Gemma 4 target without interpreting its geometry.
pub fn recognize(directory: &Directory) -> Result<(), Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| invalid("missing GGUF architecture"))?
    {
        ARCHITECTURE => Ok(()),
        architecture => Err(invalid(format!(
            "unsupported Gemma GGUF architecture {architecture:?}"
        ))),
    }
}

pub fn inspect_package(package: &Package) -> Result<ModelDefinition, Error> {
    inspect_components(
        package.target().directory(),
        package.projector().map(|projector| projector.directory()),
        package.identity(),
    )
}

/// Attention kind of one layer, from the window pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    Window,
    Full,
}

/// The attention geometry every layer of one kind shares.
struct KindGeometry {
    width: u64,
    rotary: Rotary,
    domain: HistoryDomain,
    media_rows: MediaRowAttention,
}

/// How image rows attend within their block on window layers. The released
/// configurations set `use_bidirectional_attention = "vision"` for 12B,
/// 26B-A4B and 31B, and leave it unset for E2B and E4B (transformers 5.17
/// `Gemma4Model`: with "vision", rows of one image block see each other on
/// sliding layers only; full layers stay causal; unset, every layer is
/// causal). The GGUF does not carry the key; the models without it are
/// exactly those with per-layer inputs, so that is what decides it.
fn window_media_rows(per_layer_inputs: bool) -> MediaRowAttention {
    if per_layer_inputs {
        MediaRowAttention::Causal
    } else {
        MediaRowAttention::Bidirectional
    }
}

/// NEOX rotary over a whole head: pair `p` rotates dimensions
/// `(p, p + pairs)` at `base^(-p / pairs)` for the first `rotated` pairs; the
/// other pairs have frequency zero (exact pass-through). `divisors` is the
/// stored per-pair divisor table the frequencies were derived from, checked
/// at load against the unscaled frequencies.
fn rotary(
    width: u64,
    base: f64,
    rotated: u64,
    divisors: Option<WeightDescriptor>,
) -> Result<Rotary, Error> {
    let pairs = width / 2;
    if !width.is_multiple_of(2) || rotated == 0 || rotated > pairs || base <= 1.0 {
        return Err(invalid("invalid Gemma rotary geometry"));
    }
    let bases: Vec<f64> = (0..pairs)
        .map(|pair| base.powf(-(pair as f64) / pairs as f64))
        .collect();
    Ok(Rotary::Table {
        pairs: bases
            .iter()
            .enumerate()
            .map(|(pair, base)| RotaryPair {
                frequency: if (pair as u64) < rotated { *base } else { 0.0 },
                amplitude: 1.0,
            })
            .collect(),
        divisors: divisors.map(|weight| RotaryDivisors { weight, bases }),
    })
}

/// Each layer's attention kind, from the per-layer window flags.
fn layer_kinds(m: &Metadata, layers: u64) -> Result<Vec<LayerKind>, Error> {
    let flags = m
        .flags("attention.sliding_window_pattern", layers)
        .map_err(|error| match error {
            HeaderError::MetadataType { .. } => {
                invalid("the Gemma window pattern must hold one flag per layer")
            }
            error => error.into(),
        })?;
    Ok(flags
        .into_iter()
        .map(|window| {
            if window {
                LayerKind::Window
            } else {
                LayerKind::Full
            }
        })
        .collect())
}

/// Expert geometry of a routed artifact.
struct Experts {
    count: u64,
    selected: u64,
    intermediate: u64,
}

/// Geometry every layer shares.
struct Shape {
    hidden: u64,
    heads: u64,
    epsilon: f64,
}

impl Shape {
    fn rms(&self, weight: WeightDescriptor) -> RmsNorm {
        RmsNorm {
            weight,
            epsilon: self.epsilon,
        }
    }
}

/// Inspect already-opened package components. This exists for composition
/// roots and header-only fixtures; component ownership remains with `Package`.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    recognize(directory)?;
    let m = Metadata::new(directory, ARCHITECTURE, METADATA_KEYS)?;
    let layers = m.integer("block_count")?;
    // Each layer binds distinct stored weights; reject impossible counts
    // before allocating per-layer vectors from untrusted metadata.
    if layers > directory.tensors.len() as u64 {
        return Err(invalid("Gemma layer count exceeds available weight roles"));
    }
    let shape = Shape {
        hidden: m.integer("embedding_length")?,
        heads: m.integer("attention.head_count")?,
        epsilon: m.positive_number("attention.layer_norm_rms_epsilon")?,
    };
    let hidden = shape.hidden;
    let context_limit = m.integer("context_length")?;
    let kv_heads = m.per_layer("attention.head_count_kv", layers)?;
    let feed_forward = m.per_layer("feed_forward_length", layers)?;
    let kinds = layer_kinds(&m, layers)?;
    let softcap = m.positive_number("final_logit_softcapping")?;
    let shared = m.count("attention.shared_kv_layers")?;
    let per_layer_width = m.count("embedding_length_per_layer_input")?;
    let owning = layers
        .checked_sub(shared)
        .filter(|owning| *owning > 0)
        .ok_or_else(|| invalid("Gemma shares history into more layers than it has"))?;
    let experts = match m.value("expert_count") {
        None => None,
        Some(_) => Some(Experts {
            count: m.integer("expert_count")?,
            selected: m.integer("expert_used_count")?,
            intermediate: m.integer("expert_feed_forward_length")?,
        }),
    };

    let window = {
        let width = m.integer("attention.key_length_swa")?;
        if m.integer("attention.value_length_swa")? != width
            || m.integer("rope.dimension_count_swa")? != width
        {
            return Err(invalid(
                "Gemma window layers must rotate whole heads of equal key and value width",
            ));
        }
        KindGeometry {
            width,
            rotary: rotary(width, m.positive_number("rope.freq_base_swa")?, width / 2, None)?,
            domain: HistoryDomain::Window {
                tokens: m.integer("attention.sliding_window")?,
            },
            media_rows: window_media_rows(per_layer_width > 0),
        }
    };

    let mut b = Tensors::new(directory);
    let vocabulary = b.rows("token_embd.weight")?;
    let embedding = b.bind("token_embd.weight", &[vocabulary, hidden])?;
    let output_norm = b.bind("output_norm.weight", &[hidden])?;
    let full = {
        let width = m.integer("attention.key_length")?;
        if m.integer("attention.value_length")? != width
            || m.integer("rope.dimension_count")? != width
        {
            return Err(invalid(
                "Gemma full layers must span whole heads of equal key and value width",
            ));
        }
        // The stored divisors encode the rotated fraction; the importer
        // checks them against this table.
        let divisors = b.bind("rope_freqs.weight", &[width / 2])?;
        KindGeometry {
            width,
            rotary: rotary(
                width,
                m.positive_number("rope.freq_base")?,
                width / 2 / FULL_ROTATED_PAIR_FRACTION,
                Some(divisors),
            )?,
            domain: HistoryDomain::Token,
            media_rows: MediaRowAttention::Causal,
        }
    };

    let per_layer = if per_layer_width == 0 {
        None
    } else {
        let rows = product(&[layers, per_layer_width])?;
        Some(PerLayerEntry {
            width: per_layer_width,
            layers,
            table: b.bind("per_layer_token_embd.weight", &[vocabulary, rows])?,
            table_scale: (per_layer_width as f64).sqrt(),
            projection: b.bind("per_layer_model_proj.weight", &[rows, hidden])?,
            projection_scale: 1.0 / (hidden as f64).sqrt(),
            projection_norm: shape.rms(b.bind("per_layer_proj_norm.weight", &[per_layer_width])?),
            combine_scale: std::f64::consts::FRAC_1_SQRT_2,
            media_row: PER_LAYER_MEDIA_ROW,
        })
    };

    let mut blocks = Vec::with_capacity(layers as usize);
    let mut routed_layers = 0u64;
    for layer in 0..layers {
        let index = layer as usize;
        let p = format!("blk.{layer}.");
        let kind = kinds[index];
        let geometry = match kind {
            LayerKind::Window => &window,
            LayerKind::Full => &full,
        };
        let key_value = if layer < owning {
            None
        } else {
            // A sharing layer reads the history of the last owning layer of
            // its kind.
            let source = kinds[..owning as usize]
                .iter()
                .rposition(|owner| *owner == kind)
                .ok_or_else(|| {
                    invalid(format!(
                        "Gemma layer {layer} shares no earlier layer's history"
                    ))
                })?;
            Some(source)
        };
        let attention = attention(&mut b, &shape, &p, geometry, kv_heads[index], key_value)?;
        let (input, op) = if b.contains(&format!("{p}ffn_gate_inp.weight")) {
            let experts = experts.as_ref().ok_or_else(|| {
                invalid(format!(
                    "Gemma layer {layer} routes without expert metadata"
                ))
            })?;
            routed_layers += 1;
            let dense = Branch {
                input: InputNorm::Rms(
                    shape.rms(b.bind(&format!("{p}ffn_norm.weight"), &[hidden])?),
                ),
                op: dense(&mut b, &p, hidden, feed_forward[index])?,
                output: BranchOutput::Norm(
                    shape.rms(b.bind(&format!("{p}post_ffw_norm_1.weight"), &[hidden])?),
                ),
            };
            let routed = Branch {
                input: InputNorm::Rms(
                    shape.rms(b.bind(&format!("{p}pre_ffw_norm_2.weight"), &[hidden])?),
                ),
                op: routed(&mut b, &shape, &p, experts)?,
                output: BranchOutput::Norm(
                    shape.rms(b.bind(&format!("{p}post_ffw_norm_2.weight"), &[hidden])?),
                ),
            };
            (InputNorm::None, Operator::Parallel(vec![dense, routed]))
        } else {
            (
                InputNorm::Rms(shape.rms(b.bind(&format!("{p}ffn_norm.weight"), &[hidden])?)),
                dense(&mut b, &p, hidden, feed_forward[index])?,
            )
        };
        let post_feed_forward =
            shape.rms(b.bind(&format!("{p}post_ffw_norm.weight"), &[hidden])?);
        // The layer scale multiplies the whole stream after the layer's last
        // sublayer.
        let layer_scale = b.bind(&format!("{p}layer_output_scale.weight"), &[1])?;
        let mut sublayers = vec![attention];
        match &per_layer {
            None => sublayers.push(Sublayer {
                input,
                op,
                output: OutputForm::ScaledPostNorm {
                    norm: post_feed_forward,
                    layer_scale,
                },
            }),
            Some(entry) => {
                sublayers.push(Sublayer {
                    input,
                    op,
                    output: OutputForm::PostNorm(post_feed_forward),
                });
                sublayers.push(Sublayer {
                    input: InputNorm::None,
                    op: Operator::PerLayerInput(Box::new(PerLayerInput {
                        layer,
                        width: entry.width,
                        activation: ActivationFunction::GeluTanh,
                        gate: b.bind(&format!("{p}inp_gate.weight"), &[entry.width, hidden])?,
                        projection: b.bind(&format!("{p}proj.weight"), &[hidden, entry.width])?,
                    })),
                    output: OutputForm::ScaledPostNorm {
                        norm: shape.rms(b.bind(&format!("{p}post_norm.weight"), &[hidden])?),
                        layer_scale,
                    },
                });
            }
        }
        blocks.push(Block { sublayers });
    }
    if experts.is_some() && routed_layers == 0 {
        return Err(invalid("Gemma expert metadata names no routed layer"));
    }
    b.finish()?;
    let vision = projector
        .map(|projector| describe_projector(projector, hidden))
        .transpose()?;
    let definition = ModelDefinition {
        family: FamilyId(ARCHITECTURE.into()),
        artifact_identity: identity,
        inputs: InputSemantics {
            text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
            coordinate_axes: 1,
        },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary,
            context_limit,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding: embedding.clone(),
                scale: EmbeddingScale::SqrtHidden,
                norm: None,
                per_layer,
                hash_routing: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::Rms(shape.rms(output_norm)),
                output: embedding,
                softcap: Some(softcap),
            },
        },
        head: None,
        vision,
        draft: None,
    };
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}

/// The attention sublayer of the layer at `p`. `shared` names the earlier
/// layer whose history a sharing layer reads.
fn attention(
    b: &mut Tensors,
    shape: &Shape,
    p: &str,
    geometry: &KindGeometry,
    kv_heads: u64,
    shared: Option<usize>,
) -> Result<Sublayer, Error> {
    let (hidden, width) = (shape.hidden, geometry.width);
    let head_rows = product(&[shape.heads, width])?;
    let kv_rows = product(&[kv_heads, width])?;
    let query = b.bind(&format!("{p}attn_q.weight"), &[head_rows, hidden])?;
    let query_norm =
        HeadNorm::Rms(shape.rms(b.bind(&format!("{p}attn_q_norm.weight"), &[width])?));
    let key_value = match shared {
        None => {
            let value_name = format!("{p}attn_v.weight");
            KeyValue::Owned {
                key: b.bind(&format!("{p}attn_k.weight"), &[kv_rows, hidden])?,
                // Full layers without their own value projection use their
                // raw keys as values.
                value: if b.contains(&value_name) {
                    ValueSource::Projected(b.bind(&value_name, &[kv_rows, hidden])?)
                } else {
                    ValueSource::Key
                },
                key_norm: HeadNorm::Rms(
                    shape.rms(b.bind(&format!("{p}attn_k_norm.weight"), &[width])?),
                ),
                value_norm: ValueNorm::RmsUnweighted(UnweightedRms {
                    epsilon: shape.epsilon,
                }),
                domain: geometry.domain,
            }
        }
        Some(source) => KeyValue::Shared {
            source: SublayerIndex {
                block: source as u32,
                sublayer: 0,
            },
        },
    };
    Ok(Sublayer {
        input: InputNorm::Rms(shape.rms(b.bind(&format!("{p}attn_norm.weight"), &[hidden])?)),
        op: Operator::Attention(Box::new(Attention {
            heads: shape.heads,
            kv_heads,
            width,
            query,
            gate: AttentionGate::None,
            query_norm,
            key_value,
            rotary: geometry.rotary.clone(),
            scale: 1.0,
            reads: HistoryReads::Visible,
            media_rows: geometry.media_rows,
            output: b.bind(&format!("{p}attn_output.weight"), &[hidden, head_rows])?,
        })),
        output: OutputForm::PostNorm(
            shape.rms(b.bind(&format!("{p}post_attention_norm.weight"), &[hidden])?),
        ),
    })
}

/// The GELU-tanh gated feed-forward of the layer at `p`.
fn dense(b: &mut Tensors, p: &str, hidden: u64, intermediate: u64) -> Result<Operator, Error> {
    Ok(Operator::DenseFfn(Box::new(DenseFfn {
        intermediate,
        up: FeedForwardUp::Gated {
            activation: ActivationFunction::GeluTanh,
            gate: b.bind(&format!("{p}ffn_gate.weight"), &[intermediate, hidden])?,
            up: b.bind(&format!("{p}ffn_up.weight"), &[intermediate, hidden])?,
        },
        down: b.bind(&format!("{p}ffn_down.weight"), &[hidden, intermediate])?,
    })))
}

/// The routed experts of the layer at `p`: a softmax router over
/// `RMS(residual)·scale/√D` choosing the top experts with renormalized
/// weights, each expert's fused gate‖up split by rows, and a per-expert output
/// scale.
fn routed(b: &mut Tensors, shape: &Shape, p: &str, experts: &Experts) -> Result<Operator, Error> {
    let hidden = shape.hidden;
    let fused = format!("{p}ffn_gate_up_exps.weight");
    let fused_shape = [experts.count, product(&[2, experts.intermediate])?, hidden];
    let mut half = |start: u64| {
        b.bind_transformed(
            &fused,
            &fused_shape,
            vec![ImportTransform::Rows(RowRange {
                start,
                rows: experts.intermediate,
            })],
        )
    };
    let expert_up = FeedForwardUp::Gated {
        activation: ActivationFunction::GeluTanh,
        gate: half(0)?,
        up: half(experts.intermediate)?,
    };
    let router_input = b.bind_transformed(
        &format!("{p}ffn_gate_inp.scale"),
        &[hidden],
        vec![ImportTransform::Scale {
            factor: 1.0 / (hidden as f64).sqrt(),
        }],
    )?;
    Ok(Operator::RoutedFfn(Box::new(RoutedFfn {
        experts: experts.count,
        selected: experts.selected,
        intermediate: experts.intermediate,
        router: Router {
            weight: b.bind(&format!("{p}ffn_gate_inp.weight"), &[experts.count, hidden])?,
            input: RouterInput::Residual(shape.rms(router_input)),
            score: ScoreFunction::Softmax,
            selection: ExpertSelection::TopK { bias: None },
            // The released router divides the selected probabilities by
            // their plain sum.
            normalization: RouteNormalization::Sum,
            scale: 1.0,
        },
        expert_up,
        expert_down: b.bind(
            &format!("{p}ffn_down_exps.weight"),
            &[experts.count, hidden, experts.intermediate],
        )?,
        expert_scale: Some(b.bind(&format!("{p}ffn_down_exps.scale"), &[experts.count])?),
        latent: None,
        shared: None,
    })))
}
