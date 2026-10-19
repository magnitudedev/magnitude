//! What the Qwen families share: the metadata reader, the decoder geometry of
//! the Qwen 3.5 lineage and the builders of its sublayers (attention with an
//! interleaved gate, gated delta, dense and routed feed-forward), the Qwen3-VL
//! projector description and the input adapter. A family names its own
//! architectures and composes these; numerical facts that differ between
//! families stay in the family.

pub mod inputs;
mod projector;

pub use projector::describe_projector;

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    UnsupportedReason,
};
use magnitude_family_common::architecture_keys;
use magnitude_family_contracts::{
    ActivationFunction, Attention, AttentionGate, DenseFfn, ExpertSelection, FeedForwardUp,
    GateFunction, GatedDelta, HeadNorm, HistoryDomain, HistoryReads, KeyValue, MediaRowAttention,
    Operator, RecurrentHeadMapping, RmsNorm, Rotary, RouteNormalization, RoutedFfn, Router,
    RouterInput, ScoreFunction, SharedExpert, SharedExpertGate, ValueNorm, ValueSource,
    WeightDescriptor, checked_product,
};

/// Failure to recognize or interpret a Qwen artifact: the family error,
/// which names every unsupported feature a header carries.
pub use magnitude_family_contracts::FamilyError as Error;

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
    /// Reads `family.*` keys, refusing every one outside `keys` as
    /// unrecognized metadata.
    pub fn new(directory: &'a Directory, family: &str, keys: &[&str]) -> Result<Self, Error> {
        let unknown = architecture_keys(directory, family)
            .filter(|(key, _)| !keys.contains(key))
            .map(|(_, entry)| entry.name.clone())
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            let detail = format!("unknown Qwen metadata {unknown:?}");
            return Err(Error::unsupported(
                unknown
                    .into_iter()
                    .map(|key| UnsupportedReason::UnrecognizedMetadata { key })
                    .collect(),
                detail,
            ));
        }
        Ok(Metadata {
            directory,
            prefix: format!("{family}."),
        })
    }
    /// The full metadata name of `key`.
    pub fn key(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    /// A required key: a header without it does not describe a model of
    /// the family.
    pub fn value(&self, key: &str) -> Result<&Value, Error> {
        let key = self.key(key);
        self.directory
            .value(&key)
            .ok_or_else(|| Error::invalid(format!("missing Qwen metadata {key}")))
    }
    pub fn optional(&self, key: &str) -> Option<&Value> {
        self.directory.value(&self.key(key))
    }
    /// A key present with a value outside what its reader admits.
    pub fn malformed(&self, key: &str, expected: &str) -> Error {
        let key = self.key(key);
        let detail = format!("{key} must be {expected}");
        Error::metadata(key, detail)
    }
    pub fn integer(&self, key: &str) -> Result<u64, Error> {
        self.value(key)?
            .unsigned()
            .filter(|n| *n > 0)
            .ok_or_else(|| self.malformed(key, "a positive integer"))
    }
    pub fn number(&self, key: &str) -> Result<f64, Error> {
        match self.value(key)? {
            Value::Scalar(Scalar::Float(n)) => Ok(*n),
            Value::Scalar(Scalar::Unsigned(n)) => Ok(*n as f64),
            Value::Scalar(Scalar::Signed(n)) => Ok(*n as f64),
            _ => Err(self.malformed(key, "numeric")),
        }
    }
    /// An array of nonnegative integers.
    pub fn integers(&self, key: &str) -> Result<Vec<u64>, Error> {
        let Value::Array(values) = self.value(key)? else {
            return Err(self.malformed(key, "an integer array"));
        };
        values
            .iter()
            .map(|value| match value {
                Scalar::Unsigned(n) => Some(*n),
                Scalar::Signed(n) => u64::try_from(*n).ok(),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| self.malformed(key, "an array of nonnegative integers"))
    }
}

/// The keys [`Geometry::read`] interprets in every Qwen 3.5 lineage
/// artifact. `rope.scaling.type` is read to refuse scaled rotary.
pub const GEOMETRY_KEYS: &[&str] = &[
    "embedding_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "rope.dimension_count",
    "rope.dimension_sections",
    "rope.freq_base",
    "rope.scaling.type",
    "ssm.conv_kernel",
    "ssm.group_count",
    "ssm.inner_size",
    "ssm.state_size",
    "ssm.time_step_rank",
];

/// The keys [`Geometry::read`] interprets in a dense artifact.
pub const DENSE_KEYS: &[&str] = &["feed_forward_length"];

/// The keys [`Geometry::read`] interprets in a routed artifact.
pub const ROUTED_KEYS: &[&str] = &[
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
];

/// The keys [`mixer_kinds`] interprets.
pub const MIXER_KEYS: &[&str] = &["attention.recurrent_layers", "full_attention_interval"];

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
                _ => Err(m.malformed("attention.recurrent_layers", "per-layer booleans")),
            })
            .collect(),
        _ => Err(m.malformed("attention.recurrent_layers", "one boolean per layer")),
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
        if let Some(scaling) = m
            .optional("rope.scaling.type")
            .filter(|v| v.string() != Some("none"))
        {
            let kind = scaling.string().unwrap_or("unspecified").to_owned();
            return Err(Error::unsupported(
                vec![UnsupportedReason::RopeScaling { kind }],
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

/// Rejects a projector whose output width isn't the target's embedding width: it can't feed this
/// decoder. Names (a base model's) never decide; a fine-tune keeps its base's width.
pub fn check_projector_width(projector: &Directory, hidden: u64) -> Result<(), Error> {
    match projector
        .value("clip.vision.projection_dim")
        .and_then(Value::unsigned)
    {
        Some(width) if width != hidden => Err(Error::mismatch(format!(
            "projector width {width} differs from the target's {hidden}"
        ))),
        _ => Ok(()),
    }
}
