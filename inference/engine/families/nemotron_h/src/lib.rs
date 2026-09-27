//! Nemotron-H (`nemotron_h_moe`) artifact recognition and numerical
//! interpretation.
//!
//! Every GGUF layer is one pre-normalized residual sublayer: a Mamba-2 state
//! space mixer, a NoPE attention mixer, or a feed-forward (routed "LatentMoE"
//! or dense squared-ReLU). Which one is read from the per-layer
//! `head_count_kv` and `feed_forward_length` arrays. A mixer followed by a
//! feed-forward layer forms one block; a mixer followed by another mixer is a
//! block of its own.
//!
//! Trailing `nextn_predict_layers` blocks are an embedded MTP head. It is
//! bound and shape-checked in full but not emitted: its hidden input is the
//! target's normalized final row, which the contract's head block does not
//! express, and the catalog drafts these models with DFlash instead.

mod family;

pub use family::NemotronHFamily;

use magnitude_artifacts::{
    gguf::{Directory, Value},
    PackageIdentity,
};
use magnitude_family_common::{HeaderError, Metadata, Tensors};
use magnitude_family_contracts::{
    checked_product, ActivationDType, RouteNormalization, ActivationFunction, Attention, AttentionGate, Block, Decoder,
    DefinitionError, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection,
    FamilyId, FeedForwardUp, Head, HeadBlock, HeadNorm, HistoryDomain, HistoryReads,
    ImportTransform, InputNorm, InputSemantics, KeyValue, LatentExperts, LayerNorm,
    MediaRowAttention, ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm, Rotary,
    RoutedFfn, Router, RouterInput, ScoreFunction, SharedExpert, SharedExpertGate,
    StateSpace, Sublayer, TextCoordinateSemantics, ValueNorm, ValueSource, WeightDescriptor,
};
use std::{error, fmt};

/// The GGUF architecture this family claims.
pub const ARCHITECTURE: &str = "nemotron_h_moe";

/// Every architecture metadata key the family reads. The rope keys are
/// written by the converter but unused: every attention layer is NoPE.
const KNOWN_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.layer_norm_epsilon",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.finetuned",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.inner_size",
    "ssm.time_step_rank",
    "expert_count",
    "expert_used_count",
    "expert_group_count",
    "expert_group_used_count",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "expert_shared_count",
    "expert_weights_norm",
    "expert_weights_scale",
    "moe_latent_size",
    "nextn_predict_layers",
];

/// Failure to recognize or interpret a Nemotron-H artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The artifact is not a `nemotron_h_moe` target.
    Architecture(String),
    /// Missing, malformed or unknown metadata.
    Metadata(String),
    /// A missing, misshapen or unbound tensor.
    Weight(String),
    /// A layer sequence the family cannot form into blocks.
    Structure(String),
    /// The assembled definition failed contract validation.
    Definition(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Architecture(message)
            | Self::Metadata(message)
            | Self::Weight(message)
            | Self::Structure(message)
            | Self::Definition(message) => formatter.write_str(message),
        }
    }
}

impl error::Error for Error {}

impl From<DefinitionError> for Error {
    fn from(error: DefinitionError) -> Self {
        Self::Definition(error.to_string())
    }
}

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        match error {
            HeaderError::UnknownMetadata(key) => {
                metadata_error(format!("unknown Nemotron-H metadata {key:?}"))
            }
            HeaderError::MissingWeight(name) => {
                Self::Weight(format!("missing Nemotron-H weight {name:?}"))
            }
            HeaderError::WeightShape {
                name,
                expected,
                received,
            } => Self::Weight(format!(
                "Nemotron-H weight {name:?}: expected {expected:?}, received {received:?}"
            )),
            HeaderError::UnboundWeight(name) => Self::Weight(format!(
                "Nemotron-H artifact contains unbound weight role {name:?}"
            )),
            HeaderError::Definition(error) => error.into(),
            error @ (HeaderError::NotMatrix { .. } | HeaderError::CompanionScale { .. }) => {
                Self::Weight(error.to_string())
            }
            error @ (HeaderError::MissingMetadata(_)
            | HeaderError::MetadataType { .. }
            | HeaderError::Rotary { .. }
            | HeaderError::Yarn) => metadata_error(error.to_string()),
        }
    }
}

fn metadata_error(message: impl Into<String>) -> Error {
    Error::Metadata(message.into())
}

/// Recognize a Nemotron-H target without interpreting its geometry.
pub fn recognize(directory: &Directory) -> Result<(), Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
    {
        Some(ARCHITECTURE) => Ok(()),
        Some(architecture) => Err(Error::Architecture(format!(
            "unsupported Nemotron-H GGUF architecture {architecture:?}"
        ))),
        None => Err(Error::Architecture("missing GGUF architecture".into())),
    }
}

/// What one GGUF layer is, from the per-layer metadata arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    StateSpace,
    Attention { kv_heads: u64 },
    FeedForward { intermediate: u64 },
}

impl LayerKind {
    fn of(layer: u64, kv_heads: u64, intermediate: u64) -> Result<Self, Error> {
        match (kv_heads, intermediate) {
            (0, 0) => Ok(Self::StateSpace),
            (kv_heads, 0) => Ok(Self::Attention { kv_heads }),
            (0, intermediate) => Ok(Self::FeedForward { intermediate }),
            _ => Err(Error::Structure(format!(
                "layer {layer} declares both attention heads and a feed-forward width"
            ))),
        }
    }
}

struct StateSpaceGeometry {
    inner: u64,
    heads: u64,
    head_width: u64,
    state: u64,
    groups: u64,
    convolution_width: u64,
}

struct Experts {
    count: u64,
    selected: u64,
    intermediate: u64,
    shared: Option<u64>,
    normalize: bool,
    scale: f64,
    latent: Option<u64>,
}

/// The target-wide geometry every sublayer builder reads.
struct Geometry {
    hidden: u64,
    epsilon: f64,
    attention_heads: u64,
    attention_width: u64,
    state_space: StateSpaceGeometry,
    experts: Option<Experts>,
}

impl Geometry {
    fn rms(&self, weight: WeightDescriptor) -> RmsNorm {
        RmsNorm {
            weight,
            epsilon: self.epsilon,
        }
    }

    fn attention(
        &self,
        binder: &mut Tensors,
        prefix: &str,
        kv_heads: u64,
    ) -> Result<Operator, Error> {
        let hidden = self.hidden;
        let width = self.attention_width;
        let query_rows = checked_product(&[self.attention_heads, width])?;
        let key_rows = checked_product(&[kv_heads, width])?;
        Ok(Operator::Attention(Box::new(Attention {
            heads: self.attention_heads,
            kv_heads,
            width,
            query: binder.bind(&format!("{prefix}attn_q.weight"), &[query_rows, hidden])?,
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            key_value: KeyValue::Owned {
                key: binder.bind(&format!("{prefix}attn_k.weight"), &[key_rows, hidden])?,
                value: ValueSource::Projected(
                    binder.bind(&format!("{prefix}attn_v.weight"), &[key_rows, hidden])?,
                ),
                key_norm: HeadNorm::None,
                value_norm: ValueNorm::None,
                domain: HistoryDomain::Token,
            },
            rotary: Rotary::None,
            scale: 1.0 / (width as f64).sqrt(),
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            output: binder.bind(
                &format!("{prefix}attn_output.weight"),
                &[hidden, query_rows],
            )?,
        })))
    }

    /// Mamba-2: `ssm_in` rows are `z | xBC | dt`, the contract's projection
    /// row order.
    fn state_space(&self, binder: &mut Tensors, prefix: &str) -> Result<Operator, Error> {
        let hidden = self.hidden;
        let g = &self.state_space;
        let group_channels = checked_product(&[2, g.groups, g.state])?;
        let channels = g
            .inner
            .checked_add(group_channels)
            .ok_or_else(|| metadata_error("Nemotron-H state space dimensions overflow"))?;
        let projected = channels
            .checked_add(g.inner)
            .and_then(|rows| rows.checked_add(g.heads))
            .ok_or_else(|| metadata_error("Nemotron-H state space dimensions overflow"))?;
        let norm_group = g.inner / g.groups;
        Ok(Operator::StateSpace(Box::new(StateSpace {
            heads: g.heads,
            head_width: g.head_width,
            state: g.state,
            groups: g.groups,
            convolution_width: g.convolution_width,
            projection: binder.bind(&format!("{prefix}ssm_in.weight"), &[projected, hidden])?,
            convolution: binder.bind(
                &format!("{prefix}ssm_conv1d.weight"),
                &[channels, g.convolution_width],
            )?,
            convolution_bias: binder.bind(&format!("{prefix}ssm_conv1d.bias"), &[channels])?,
            time_bias: binder.bind(&format!("{prefix}ssm_dt.bias"), &[g.heads])?,
            decay: binder.bind_transformed(
                &format!("{prefix}ssm_a"),
                &[g.heads, 1],
                vec![ImportTransform::Flatten],
            )?,
            skip: binder.bind_transformed(
                &format!("{prefix}ssm_d"),
                &[g.heads, 1],
                vec![ImportTransform::Flatten],
            )?,
            norm: self.rms(binder.bind_transformed(
                &format!("{prefix}ssm_norm.weight"),
                &[g.groups, norm_group],
                vec![ImportTransform::Flatten],
            )?),
            norm_group,
            output: binder.bind(&format!("{prefix}ssm_out.weight"), &[hidden, g.inner])?,
        })))
    }

    /// A routed feed-forward when the layer has a router, else a dense one.
    /// Both use non-gated squared-ReLU expansions.
    fn feed_forward(
        &self,
        binder: &mut Tensors,
        prefix: &str,
        intermediate: u64,
    ) -> Result<Operator, Error> {
        let hidden = self.hidden;
        let router = format!("{prefix}ffn_gate_inp.weight");
        if !binder.contains(&router) {
            return Ok(Operator::DenseFfn(Box::new(DenseFfn {
                intermediate,
                up: FeedForwardUp::Plain {
                    activation: ActivationFunction::ReluSquared,
                    up: binder
                        .bind_scaled(&format!("{prefix}ffn_up.weight"), &[intermediate, hidden])?,
                },
                down: binder
                    .bind_scaled(&format!("{prefix}ffn_down.weight"), &[hidden, intermediate])?,
            })));
        }
        let e = self.experts.as_ref().ok_or_else(|| {
            metadata_error(format!("routed layer {prefix} without expert metadata"))
        })?;
        if intermediate != e.intermediate {
            return Err(Error::Structure(format!(
                "routed layer {prefix} declares width {intermediate}, experts have {}",
                e.intermediate
            )));
        }
        let expert_hidden = e.latent.unwrap_or(hidden);
        let latent = e
            .latent
            .map(|width| -> Result<LatentExperts, Error> {
                Ok(LatentExperts {
                    width,
                    down: binder
                        .bind(&format!("{prefix}ffn_latent_down.weight"), &[width, hidden])?,
                    up: binder
                        .bind(&format!("{prefix}ffn_latent_up.weight"), &[hidden, width])?,
                })
            })
            .transpose()?;
        let shared = e
            .shared
            .map(|shared| -> Result<SharedExpert, Error> {
                Ok(SharedExpert {
                    intermediate: shared,
                    up: FeedForwardUp::Plain {
                        activation: ActivationFunction::ReluSquared,
                        up: binder
                            .bind_scaled(&format!("{prefix}ffn_up_shexp.weight"), &[shared, hidden])?,
                    },
                    down: binder
                        .bind_scaled(&format!("{prefix}ffn_down_shexp.weight"), &[hidden, shared])?,
                    gate: SharedExpertGate::None,
                })
            })
            .transpose()?;
        Ok(Operator::RoutedFfn(Box::new(RoutedFfn {
            experts: e.count,
            selected: e.selected,
            intermediate,
            router: Router {
                weight: binder.bind(&router, &[e.count, hidden])?,
                input: RouterInput::Operator,
                score: ScoreFunction::Sigmoid,
                selection: ExpertSelection::TopK {
                    bias: Some(binder.bind(&format!("{prefix}exp_probs_b.bias"), &[e.count])?),
                },
                // The released model divides by `sum + 1e-20` (llama.cpp
                // clamps the sum at 6.1e-5 instead); the model definition
                // governs.
                normalization: if e.normalize {
                    RouteNormalization::SumPlusEpsilon(1e-20)
                } else {
                    RouteNormalization::None
                },
                scale: e.scale,
            },
            expert_up: FeedForwardUp::Plain {
                activation: ActivationFunction::ReluSquared,
                up: binder.bind_scaled(
                    &format!("{prefix}ffn_up_exps.weight"),
                    &[e.count, intermediate, expert_hidden],
                )?,
            },
            expert_down: binder.bind_scaled(
                &format!("{prefix}ffn_down_exps.weight"),
                &[e.count, expert_hidden, intermediate],
            )?,
            expert_scale: None,
            latent,
            shared,
        })))
    }

    fn sublayer(
        &self,
        binder: &mut Tensors,
        layer: u64,
        kind: LayerKind,
    ) -> Result<Sublayer, Error> {
        let prefix = format!("blk.{layer}.");
        let op = match kind {
            LayerKind::StateSpace => self.state_space(binder, &prefix)?,
            LayerKind::Attention { kv_heads } => self.attention(binder, &prefix, kv_heads)?,
            LayerKind::FeedForward { intermediate } => {
                self.feed_forward(binder, &prefix, intermediate)?
            }
        };
        Ok(Sublayer {
            input: InputNorm::Rms(
                self.rms(binder.bind(&format!("{prefix}attn_norm.weight"), &[self.hidden])?),
            ),
            op,
            output: OutputForm::Residual,
        })
    }
}

/// Group layers into blocks: each mixer takes the feed-forward layer that
/// follows it, and a mixer followed by another mixer stands alone.
fn block_layers(kinds: &[LayerKind]) -> Result<Vec<Vec<(u64, LayerKind)>>, Error> {
    let mut blocks = Vec::new();
    let mut layers = (0u64..).zip(kinds.iter().copied()).peekable();
    while let Some((layer, kind)) = layers.next() {
        if matches!(kind, LayerKind::FeedForward { .. }) {
            return Err(Error::Structure(format!(
                "feed-forward layer {layer} follows no mixer layer"
            )));
        }
        let mut block = vec![(layer, kind)];
        block.extend(layers.next_if(|(_, kind)| matches!(kind, LayerKind::FeedForward { .. })));
        blocks.push(block);
    }
    Ok(blocks)
}

/// Inspect a Nemotron-H target from its headers.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    recognize(directory)?;
    if projector.is_some() {
        return Err(Error::Structure(
            "Nemotron-H has no vision projector".into(),
        ));
    }
    let m = Metadata::new(directory, ARCHITECTURE, KNOWN_KEYS)?;
    let block_count = m.integer("block_count")?;
    let head_layers = m.optional_count("nextn_predict_layers")?.unwrap_or(0);
    if head_layers >= block_count {
        return Err(metadata_error(
            "nextn_predict_layers must be below block count",
        ));
    }
    // Every layer binds distinct stored weights; reject impossible counts
    // before allocating per-layer vectors from untrusted metadata.
    if block_count > directory.tensors.len() as u64 {
        return Err(metadata_error("Nemotron-H block count exceeds its weights"));
    }
    let kv_heads = m.layer_counts("attention.head_count_kv", block_count)?;
    let intermediates = m.layer_counts("feed_forward_length", block_count)?;

    let hidden = m.integer("embedding_length")?;
    let attention_width = m.integer("attention.key_length")?;
    if m.integer("attention.value_length")? != attention_width {
        return Err(metadata_error(
            "Nemotron-H attention key and value widths differ",
        ));
    }
    let inner = m.integer("ssm.inner_size")?;
    let heads = m.integer("ssm.time_step_rank")?;
    let groups = m.integer("ssm.group_count")?;
    if !inner.is_multiple_of(heads) || !heads.is_multiple_of(groups) {
        return Err(metadata_error(
            "Nemotron-H state space heads do not divide its inner width into groups",
        ));
    }
    let experts = match m.value("expert_count") {
        None => None,
        Some(_) => {
            if m.integer("expert_group_count")? != 1 || m.integer("expert_group_used_count")? != 1 {
                return Err(metadata_error(
                    "group-limited Nemotron-H expert routing is not supported",
                ));
            }
            let shared = match m.count("expert_shared_count")? {
                0 => None,
                1 => Some(m.integer("expert_shared_feed_forward_length")?),
                _ => {
                    return Err(metadata_error(
                        "Nemotron-H supports at most one shared expert",
                    ))
                }
            };
            Some(Experts {
                count: m.integer("expert_count")?,
                selected: m.integer("expert_used_count")?,
                intermediate: m.integer("expert_feed_forward_length")?,
                shared,
                normalize: m.flag("expert_weights_norm")?,
                scale: m.number("expert_weights_scale")?,
                latent: m.optional_integer("moe_latent_size")?,
            })
        }
    };
    let geometry = Geometry {
        hidden,
        epsilon: m.number("attention.layer_norm_rms_epsilon")?,
        attention_heads: m.integer("attention.head_count")?,
        attention_width,
        state_space: StateSpaceGeometry {
            inner,
            heads,
            head_width: inner / heads,
            state: m.integer("ssm.state_size")?,
            groups,
            convolution_width: m.integer("ssm.conv_kernel")?,
        },
        experts,
    };

    let mut binder = Tensors::new(directory);
    let vocabulary = binder.rows("token_embd.weight")?;
    if m.integer("vocab_size")? != vocabulary {
        return Err(metadata_error(
            "Nemotron-H vocabulary size differs from its embedding",
        ));
    }
    let embedding = binder.bind("token_embd.weight", &[vocabulary, hidden])?;
    let output_name = if binder.contains("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let output = binder.bind_scaled(output_name, &[vocabulary, hidden])?;
    let output_norm = binder.bind("output_norm.weight", &[hidden])?;

    let target_layers = block_count - head_layers;
    let kinds = (0..target_layers)
        .map(|layer| {
            let index = layer as usize;
            LayerKind::of(layer, kv_heads[index], intermediates[index])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let blocks = block_layers(&kinds)?
        .into_iter()
        .map(|layers| {
            Ok(Block {
                sublayers: layers
                    .into_iter()
                    .map(|(layer, kind)| geometry.sublayer(&mut binder, layer, kind))
                    .collect::<Result<_, Error>>()?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let decoder = Decoder {
        activation_dtype: ActivationDType::BF16,
        hidden,
        vocabulary,
        context_limit: m.integer("context_length")?,
        residual: ResidualForm::Single,
        entry: EntryForm {
            embedding,
            scale: EmbeddingScale::Unit,
            norm: None,
            per_layer: None,
            hash_routing: None,
        },
        blocks,
        exit: ExitForm {
            norm: ExitNorm::Rms(geometry.rms(output_norm)),
            output,
            softcap: None,
        },
    };
    let inputs = InputSemantics {
        text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
        coordinate_axes: 1,
    };

    if head_layers > 0 {
        let head = Head {
            blocks: (target_layers..block_count)
                .map(|layer| {
                    let index = layer as usize;
                    let (kv_heads, intermediate) = (kv_heads[index], intermediates[index]);
                    if kv_heads == 0 || intermediate == 0 {
                        return Err(Error::Structure(format!(
                            "prediction layer {layer} must declare attention heads and a feed-forward width"
                        )));
                    }
                    let epsilon = m.number("attention.layer_norm_epsilon")?;
                    head_block(&geometry, &mut binder, layer, kv_heads, intermediate, epsilon)
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        // Bound and checked, then deliberately not emitted (module docs).
        head.validate(&decoder, inputs.coordinate_axes)?;
    }
    binder.finish()?;
    let definition = ModelDefinition {
        family: FamilyId(ARCHITECTURE.into()),
        artifact_identity: identity,
        inputs,
        decoder,
        head: None,
        vision: None,
        draft: None,
    };
    definition.validate()?;
    Ok(definition)
}

/// One MTP block: attention then feed-forward under the `nextn` combine,
/// finished by a mean-subtracted LayerNorm.
fn head_block(
    geometry: &Geometry,
    binder: &mut Tensors,
    layer: u64,
    kv_heads: u64,
    intermediate: u64,
    layer_norm_epsilon: f64,
) -> Result<HeadBlock, Error> {
    let hidden = geometry.hidden;
    let prefix = format!("blk.{layer}.");
    let mut weight = |name: &str, shape: &[u64]| binder.bind(&format!("{prefix}{name}"), shape);
    let embedding_norm = geometry.rms(weight("nextn.enorm.weight", &[hidden])?);
    let hidden_norm = geometry.rms(weight("nextn.hnorm.weight", &[hidden])?);
    let combine = weight(
        "nextn.eh_proj.weight",
        &[hidden, checked_product(&[2, hidden])?],
    )?;
    let attention_norm = geometry.rms(weight("attn_norm.weight", &[hidden])?);
    let feed_forward_norm = geometry.rms(weight("post_attention_norm.weight", &[hidden])?);
    let output_norm = ExitNorm::Layer(LayerNorm {
        weight: weight("nextn.shared_head_norm.weight", &[hidden])?,
        epsilon: layer_norm_epsilon,
    });
    Ok(HeadBlock {
        embedding_norm,
        hidden_norm,
        combine,
        block: Block {
            sublayers: vec![
                Sublayer {
                    input: InputNorm::Rms(attention_norm),
                    op: geometry.attention(binder, &prefix, kv_heads)?,
                    output: OutputForm::Residual,
                },
                Sublayer {
                    input: InputNorm::Rms(feed_forward_norm),
                    op: geometry.feed_forward(binder, &prefix, intermediate)?,
                    output: OutputForm::Residual,
                },
            ],
        },
        output_norm,
    })
}
