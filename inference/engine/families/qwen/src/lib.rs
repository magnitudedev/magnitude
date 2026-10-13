//! What the Qwen families share: the metadata reader, the decoder geometry of
//! the Qwen 3.5 lineage and the builders of its sublayers (attention with an
//! interleaved gate, gated delta, dense and routed feed-forward), the Qwen3-VL
//! projector description and the input adapter. A family names its own
//! architectures and composes these; numerical facts that differ between
//! families stay in the family.

pub mod inputs;
mod projector;

pub use projector::describe_projector;

use magnitude_artifacts::gguf::{Directory, Scalar, Value};
use magnitude_family_common::HeaderError;
use magnitude_family_contracts::{
    ActivationFunction, Attention, AttentionGate, DenseFfn, ExpertSelection, FeedForwardUp,
    GateFunction, GatedDelta, HeadNorm, HistoryDomain, HistoryReads, KeyValue, MediaRowAttention,
    Operator, RecurrentHeadMapping, RmsNorm, Rotary, RouteNormalization, RoutedFfn, Router,
    RouterInput, ScoreFunction, SharedExpert, SharedExpertGate, ValueNorm, ValueSource,
    WeightDescriptor, checked_product,
};
use std::{error, fmt};

/// Failure to recognize or interpret a Qwen artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for Error {}

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        Error::invalid(error.to_string())
    }
}

pub fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(|error| Error::invalid(error.to_string()))
}

/// Binds a block's weight by its name within the block.
pub type Weight<'a> = dyn FnMut(&str, &[u64]) -> Result<WeightDescriptor, Error> + 'a;

/// Metadata of one family, read under its `{family}.` prefix.
pub struct Metadata<'a> {
    pub directory: &'a Directory,
    pub prefix: String,
}

impl<'a> Metadata<'a> {
    pub fn new(directory: &'a Directory, family: &str) -> Self {
        Metadata {
            directory,
            prefix: format!("{family}."),
        }
    }
    pub fn value(&self, key: &str) -> Result<&Value, Error> {
        self.directory
            .value(&format!("{}{key}", self.prefix))
            .ok_or_else(|| Error::invalid(format!("missing Qwen metadata {}{key}", self.prefix)))
    }
    pub fn optional(&self, key: &str) -> Option<&Value> {
        self.directory.value(&format!("{}{key}", self.prefix))
    }
    pub fn integer(&self, key: &str) -> Result<u64, Error> {
        self.value(key)?
            .unsigned()
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                Error::invalid(format!("{}{key} must be a positive integer", self.prefix))
            })
    }
    pub fn number(&self, key: &str) -> Result<f64, Error> {
        match self.value(key)? {
            Value::Scalar(Scalar::Float(n)) => Ok(*n),
            Value::Scalar(Scalar::Unsigned(n)) => Ok(*n as f64),
            Value::Scalar(Scalar::Signed(n)) => Ok(*n as f64),
            _ => Err(Error::invalid(format!(
                "{}{key} must be numeric",
                self.prefix
            ))),
        }
    }
    /// An array of nonnegative integers.
    pub fn integers(&self, key: &str) -> Result<Vec<u64>, Error> {
        let Value::Array(values) = self.value(key)? else {
            return Err(Error::invalid(format!(
                "{}{key} must be an integer array",
                self.prefix
            )));
        };
        values
            .iter()
            .map(|value| match value {
                Scalar::Unsigned(n) => Some(*n),
                Scalar::Signed(n) => u64::try_from(*n).ok(),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{}{key} must hold nonnegative integers",
                    self.prefix
                ))
            })
    }
}

/// The mixer of one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    Attention,
    Recurrent,
}

/// Every layer's mixer: per-layer recurrent flags, or one attention layer
/// per `full_attention_interval`.
pub fn mixer_kinds(m: &Metadata, layer_count: u64) -> Result<Vec<MixerKind>, Error> {
    match m.optional("attention.recurrent_layers") {
        None => {
            let interval = m.integer("full_attention_interval")?;
            Ok((0..layer_count)
                .map(|i| {
                    if (i + 1) % interval == 0 {
                        MixerKind::Attention
                    } else {
                        MixerKind::Recurrent
                    }
                })
                .collect())
        }
        Some(Value::Array(flags)) if flags.len() as u64 == layer_count => flags
            .iter()
            .map(|v| match v {
                Scalar::Bool(true) => Ok(MixerKind::Recurrent),
                Scalar::Bool(false) => Ok(MixerKind::Attention),
                _ => Err(Error::invalid("invalid per-layer recurrent flags")),
            })
            .collect(),
        _ => Err(Error::invalid("invalid per-layer recurrent flags")),
    }
}

/// Expert widths of a routed artifact.
#[derive(Clone, Debug)]
pub struct Experts {
    pub count: u64,
    pub selected: u64,
    pub intermediate: u64,
    pub shared_intermediate: u64,
}

/// The decoder geometry of the Qwen 3.5 lineage.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub hidden: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub attention_width: u64,
    pub epsilon: f64,
    pub rotary: Rotary,
    pub convolution_width: u64,
    pub recurrent_key_heads: u64,
    pub recurrent_value_heads: u64,
    pub recurrent_width: u64,
    pub recurrent_inner: u64,
    pub recurrent_channels: u64,
    pub experts: Option<Experts>,
    /// The dense feed-forward width, or the expert width when routed.
    pub intermediate: u64,
}

impl Geometry {
    pub fn read(m: &Metadata, routed: bool) -> Result<Self, Error> {
        if m.optional("rope.scaling.type")
            .is_some_and(|v| v.string() != Some("none"))
        {
            return Err(Error::invalid(
                "scaled Qwen rotary definitions are not yet qualified",
            ));
        }
        let Value::Array(sections) = m.value("rope.dimension_sections")? else {
            return Err(Error::invalid(
                "Qwen rotary sections must contain four integers",
            ));
        };
        let sections = sections
            .iter()
            .map(|v| match v {
                Scalar::Unsigned(n) => Some(*n),
                Scalar::Signed(n) => u64::try_from(*n).ok(),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|v| <[u64; 4]>::try_from(v).ok())
            .ok_or_else(|| {
                Error::invalid("Qwen rotary sections must contain four nonnegative integers")
            })?;
        if sections[3] != 0 {
            return Err(Error::invalid("Qwen rotary fourth section must be zero"));
        }
        let experts = if routed {
            Some(Experts {
                count: m.integer("expert_count")?,
                selected: m.integer("expert_used_count")?,
                intermediate: m.integer("expert_feed_forward_length")?,
                shared_intermediate: m.integer("expert_shared_feed_forward_length")?,
            })
        } else {
            None
        };
        let hidden = m.integer("embedding_length")?;
        let intermediate = match &experts {
            Some(experts) => experts.intermediate,
            None => m.integer("feed_forward_length")?,
        };
        let attention_heads = m.integer("attention.head_count")?;
        let kv_heads = m.integer("attention.head_count_kv")?;
        let attention_width = m.integer("attention.key_length")?;
        let recurrent_key_heads = m.integer("ssm.group_count")?;
        let recurrent_value_heads = m.integer("ssm.time_step_rank")?;
        let recurrent_width = m.integer("ssm.state_size")?;
        let recurrent_inner = product(&[recurrent_value_heads, recurrent_width])?;
        if m.integer("attention.value_length")? != attention_width
            || m.integer("ssm.inner_size")? != recurrent_inner
        {
            return Err(Error::invalid(
                "Qwen attention/recurrent inner size differs from head geometry",
            ));
        }
        let epsilon = m.number("attention.layer_norm_rms_epsilon")?;
        let rotary = Rotary::Interleaved {
            width: m.integer("rope.dimension_count")?,
            base: m.number("rope.freq_base")?,
            sections: sections.into(),
            axis_pattern: vec![0, 1, 2],
        };
        let convolution_width = m.integer("ssm.conv_kernel")?;
        let recurrent_channels = product(&[
            recurrent_key_heads
                .checked_mul(2)
                .and_then(|heads| heads.checked_add(recurrent_value_heads))
                .ok_or_else(|| Error::invalid("Qwen recurrent dimensions overflow"))?,
            recurrent_width,
        ])?;
        Ok(Self {
            hidden,
            attention_heads,
            kv_heads,
            attention_width,
            epsilon,
            rotary,
            convolution_width,
            recurrent_key_heads,
            recurrent_value_heads,
            recurrent_width,
            recurrent_inner,
            recurrent_channels,
            experts,
            intermediate,
        })
    }

    pub fn rms(&self, weight: WeightDescriptor) -> RmsNorm {
        RmsNorm {
            weight,
            epsilon: self.epsilon,
        }
    }

    /// Attention with a sigmoid gate interleaved in the query rows, head
    /// norms on queries and keys, and its own token history.
    pub fn attention(&self, weight: &mut Weight) -> Result<Attention, Error> {
        let (heads, kv_heads, width, hidden) = (
            self.attention_heads,
            self.kv_heads,
            self.attention_width,
            self.hidden,
        );
        Ok(Attention {
            heads,
            kv_heads,
            width,
            query: weight("attn_q.weight", &[product(&[2, heads, width])?, hidden])?,
            gate: AttentionGate::Interleaved {
                function: GateFunction::Sigmoid,
            },
            query_norm: HeadNorm::Rms(self.rms(weight("attn_q_norm.weight", &[width])?)),
            key_value: KeyValue::Owned {
                key: weight("attn_k.weight", &[product(&[kv_heads, width])?, hidden])?,
                value: ValueSource::Projected(weight(
                    "attn_v.weight",
                    &[product(&[kv_heads, width])?, hidden],
                )?),
                key_norm: HeadNorm::Rms(self.rms(weight("attn_k_norm.weight", &[width])?)),
                value_norm: ValueNorm::None,
                domain: HistoryDomain::Token,
            },
            rotary: self.rotary.clone(),
            scale: 1.0 / (width as f64).sqrt(),
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            output: weight("attn_output.weight", &[hidden, product(&[heads, width])?])?,
        })
    }

    /// Gated DeltaNet with tiled heads.
    pub fn gated_delta(
        &self,
        weight: &mut Weight,
        output_gate: GateFunction,
    ) -> Result<GatedDelta, Error> {
        let (channels, inner, value_heads, hidden) = (
            self.recurrent_channels,
            self.recurrent_inner,
            self.recurrent_value_heads,
            self.hidden,
        );
        Ok(GatedDelta {
            convolution_width: self.convolution_width,
            key_heads: self.recurrent_key_heads,
            value_heads,
            width: self.recurrent_width,
            head_mapping: RecurrentHeadMapping::Tiled,
            query_key_value: weight("attn_qkv.weight", &[channels, hidden])?,
            gate: weight("attn_gate.weight", &[inner, hidden])?,
            alpha: weight("ssm_alpha.weight", &[value_heads, hidden])?,
            beta: weight("ssm_beta.weight", &[value_heads, hidden])?,
            convolution: weight("ssm_conv1d.weight", &[channels, self.convolution_width])?,
            decay: weight("ssm_a", &[value_heads])?,
            time_bias: weight("ssm_dt.bias", &[value_heads])?,
            norm: self.rms(weight("ssm_norm.weight", &[self.recurrent_width])?),
            output_gate,
            output: weight("ssm_out.weight", &[hidden, inner])?,
        })
    }

    /// The routed feed-forward with a sigmoid-gated shared expert, or the
    /// dense one of `intermediate` width.
    pub fn feed_forward(&self, weight: &mut Weight, intermediate: u64) -> Result<Operator, Error> {
        let hidden = self.hidden;
        Ok(if let Some(e) = &self.experts {
            Operator::RoutedFfn(Box::new(RoutedFfn {
                experts: e.count,
                selected: e.selected,
                intermediate: e.intermediate,
                router: Router {
                    weight: weight("ffn_gate_inp.weight", &[e.count, hidden])?,
                    input: RouterInput::Operator,
                    score: ScoreFunction::Softmax,
                    selection: ExpertSelection::TopK { bias: None },
                    normalization: RouteNormalization::Sum,
                    scale: 1.0,
                },
                expert_up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    gate: weight("ffn_gate_exps.weight", &[e.count, e.intermediate, hidden])?,
                    up: weight("ffn_up_exps.weight", &[e.count, e.intermediate, hidden])?,
                },
                expert_down: weight("ffn_down_exps.weight", &[e.count, hidden, e.intermediate])?,
                expert_scale: None,
                latent: None,
                shared: Some(SharedExpert {
                    intermediate: e.shared_intermediate,
                    up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        gate: weight("ffn_gate_shexp.weight", &[e.shared_intermediate, hidden])?,
                        up: weight("ffn_up_shexp.weight", &[e.shared_intermediate, hidden])?,
                    },
                    down: weight("ffn_down_shexp.weight", &[hidden, e.shared_intermediate])?,
                    gate: SharedExpertGate::Sigmoid(weight(
                        "ffn_gate_inp_shexp.weight",
                        &[hidden],
                    )?),
                }),
            }))
        } else {
            Operator::DenseFfn(Box::new(DenseFfn {
                intermediate,
                up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    gate: weight("ffn_gate.weight", &[intermediate, hidden])?,
                    up: weight("ffn_up.weight", &[intermediate, hidden])?,
                },
                down: weight("ffn_down.weight", &[hidden, intermediate])?,
            }))
        })
    }
}

/// Rejects a projector whose base model differs from the target's. A derived
/// target (a quantization, a fine-tune) names its base model exactly as the
/// projector does; an original model is its own base.
pub fn check_projector_base(directory: &Directory, projector: &Directory) -> Result<(), Error> {
    let target_base = directory
        .value("general.base_model.0.name")
        .and_then(Value::string)
        .or_else(|| directory.value("general.name").and_then(Value::string));
    let projector_base = projector
        .value("general.base_model.0.name")
        .and_then(Value::string);
    if target_base
        .zip(projector_base)
        .is_some_and(|(target, projected)| target != projected)
    {
        return Err(Error::invalid("projector base model differs from target"));
    }
    Ok(())
}
