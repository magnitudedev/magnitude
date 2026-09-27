//! The decoder as blocks of sublayers over a residual form (model-family plan
//! §3.1).
//!
//! A decoder is an entry form, an ordered list of blocks and an exit form over
//! one residual form. A block is an ordered list of sublayers; a sublayer
//! normalizes its residual input, applies one operator and writes its output
//! back through an output form. Every operator carries its own geometry and
//! its bound weights, so a definition cannot pair a geometry with weights of a
//! different operator.
//!
//! Some forms exist only as interface points for deferred families (plan §9):
//! hyper-connection residuals, latent attention, block-rate and gathered
//! history, hash routing and sqrt-softplus scores. They are legal
//! descriptions; [`ModelDefinition::deferred_forms`](crate::ModelDefinition::deferred_forms)
//! names them so an executor rejects them with a typed error.

use crate::{checked_product, expect_shape, DefinitionError, WeightDescriptor};
use serde::{Deserialize, Serialize};

/// Root-mean-square normalization with a stored per-channel multiplier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RmsNorm {
    pub weight: WeightDescriptor,
    pub epsilon: f64,
}

/// Mean-subtracted normalization with a stored multiplier and no bias.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNorm {
    pub weight: WeightDescriptor,
    pub epsilon: f64,
}

/// How the residual stream is carried between sublayers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResidualForm {
    /// One F32 residual row per token.
    Single,
    /// Several residual streams mixed around every sublayer. Deferred (§9).
    HyperConnections(HyperConnections),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperConnections {
    pub streams: u64,
    pub mixing: HyperConnectionMixing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HyperConnectionMixing {
    LowRankGated { rank: u64 },
    Sinkhorn { iterations: u64 },
}

/// How token rows enter the residual stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntryForm {
    /// `[vocabulary, hidden]` token table.
    pub embedding: WeightDescriptor,
    /// Multiplier applied to text rows (media rows enter unscaled).
    pub scale: EmbeddingScale,
    /// Unweighted normalization of every entering row.
    pub norm: Option<UnweightedRms>,
    /// Per-layer inputs gathered once per token (Gemma PLE).
    pub per_layer: Option<PerLayerEntry>,
    /// Token-keyed expert table for hash-routed layers. Deferred (§9).
    pub hash_routing: Option<HashRoutingTable>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingScale {
    Unit,
    /// `sqrt(hidden)`, computed in F32.
    SqrtHidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnweightedRms {
    pub epsilon: f64,
}

/// The per-layer input table: for layer `l` and token `t`,
/// `ple[l] = (RMS(projection·h0)[l] + table[t][l]·table_scale) · combine_scale`,
/// where `projection·h0` is scaled by `projection_scale` and normalized per
/// `width`-wide chunk with one shared weight. `h0` is the scaled entry row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerLayerEntry {
    /// Channels per layer (`P`).
    pub width: u64,
    /// Layers the table covers (`L`).
    pub layers: u64,
    /// `[vocabulary, L·P]` host-resident gathered table.
    pub table: WeightDescriptor,
    pub table_scale: f64,
    /// `[L·P, hidden]`.
    pub projection: WeightDescriptor,
    pub projection_scale: f64,
    /// `[P]`, shared by every chunk.
    pub projection_norm: RmsNorm,
    pub combine_scale: f64,
    /// The table row media rows read in place of a token.
    pub media_row: u64,
}

/// Deferred (§9): token-keyed expert selection for the first routed layers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HashRoutingTable {
    pub table: WeightDescriptor,
}

/// How the final residual becomes logits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExitForm {
    pub norm: ExitNorm,
    /// `[vocabulary, hidden]`; tied when it names the entry embedding.
    pub output: WeightDescriptor,
    /// `cap · tanh(z / cap)` in F32 on the logits.
    pub softcap: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExitNorm {
    Rms(RmsNorm),
    Layer(LayerNorm),
    /// Collapses hyper-connection streams before readout. Deferred (§9).
    HyperConnectionHead(HyperConnectionWeights),
}

impl ExitNorm {
    pub fn weight(&self) -> &WeightDescriptor {
        match self {
            Self::Rms(norm) => &norm.weight,
            Self::Layer(norm) => &norm.weight,
            Self::HyperConnectionHead(weights) => &weights.mix,
        }
    }
}

/// Deferred (§9): the mixing weights of one hyper-connection site.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperConnectionWeights {
    pub mix: WeightDescriptor,
}

/// One decoder layer: usually `[mixer, feed-forward]`, possibly a lone mixer
/// or with further sublayers (Gemma per-layer input).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub sublayers: Vec<Sublayer>,
}

/// `residual ← output(residual, op(input(residual)))`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sublayer {
    pub input: InputNorm,
    pub op: Operator,
    pub output: OutputForm,
}

/// Structural address of one sublayer inside a block list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SublayerIndex {
    pub block: u32,
    pub sublayer: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InputNorm {
    Rms(RmsNorm),
    RmsUnweighted(UnweightedRms),
    /// Mixes hyper-connection streams into the sublayer input. Deferred (§9).
    HyperConnectionMix(HyperConnectionWeights),
    /// The operator reads the raw residual.
    None,
}

impl InputNorm {
    pub fn epsilon(&self) -> Option<f64> {
        match self {
            Self::Rms(norm) => Some(norm.epsilon),
            Self::RmsUnweighted(norm) => Some(norm.epsilon),
            Self::HyperConnectionMix(_) | Self::None => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutputForm {
    /// `residual + y`.
    Residual,
    /// `residual + RMS(y)·w`.
    PostNorm(RmsNorm),
    /// `(residual + RMS(y)·w) · layer_scale`, scaling the whole stream.
    ScaledPostNorm {
        norm: RmsNorm,
        /// `[1]`.
        layer_scale: WeightDescriptor,
    },
}

/// The operator of one sublayer, each carrying its geometry and weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    Attention(Box<Attention>),
    /// Deferred (§9): latent multi-query attention.
    LatentAttention(Box<LatentAttention>),
    GatedDelta(Box<GatedDelta>),
    ShortConv(Box<ShortConv>),
    StateSpace(Box<StateSpace>),
    DenseFfn(Box<DenseFfn>),
    RoutedFfn(Box<RoutedFfn>),
    PerLayerInput(Box<PerLayerInput>),
    /// Independent branches over the same residual input whose outputs sum.
    Parallel(Vec<Branch>),
}

impl Operator {
    /// Stable operator name, for diagnostics and typed rejections.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Attention(_) => "attention",
            Self::LatentAttention(_) => "latent attention",
            Self::GatedDelta(_) => "gated delta",
            Self::ShortConv(_) => "short convolution",
            Self::StateSpace(_) => "state space",
            Self::DenseFfn(_) => "dense feed-forward",
            Self::RoutedFfn(_) => "routed feed-forward",
            Self::PerLayerInput(_) => "per-layer input",
            Self::Parallel(_) => "parallel branches",
        }
    }
}

/// One branch of a [`Operator::Parallel`] sublayer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Branch {
    pub input: InputNorm,
    pub op: Operator,
    pub output: BranchOutput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BranchOutput {
    Plain,
    Norm(RmsNorm),
}

// --- attention ---------------------------------------------------------------

/// Softmax attention over history spans. `W` is the head width of queries,
/// keys and values alike.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attention {
    pub heads: u64,
    pub kv_heads: u64,
    pub width: u64,
    /// `[heads·W, hidden]`, or `[2·heads·W, hidden]` when the gate is
    /// interleaved with each head's query rows.
    pub query: WeightDescriptor,
    pub gate: AttentionGate,
    pub query_norm: HeadNorm,
    pub key_value: KeyValue,
    pub rotary: Rotary,
    /// Multiplier on `q·k` before the softmax.
    pub scale: f64,
    pub reads: HistoryReads,
    /// Whether media rows of one span see each other.
    pub media_rows: MediaRowAttention,
    /// `[hidden, heads·W]`.
    pub output: WeightDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateFunction {
    Sigmoid,
    Softplus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateGranularity {
    /// One gate value per attention output element.
    Element,
    /// One gate value per head, broadcast over its width.
    Head,
}

/// Multiplicative gate on the attention output before the output projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttentionGate {
    None,
    /// Per-element gate rows follow each head's query rows in `query`.
    Interleaved { function: GateFunction },
    /// Gate from its own projection: `[heads·W, hidden]` per element, or
    /// `[heads, hidden]` per head.
    Separate {
        weight: WeightDescriptor,
        function: GateFunction,
        granularity: GateGranularity,
    },
}

/// Per-head normalization over the head width with one shared weight.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HeadNorm {
    /// No normalization (unit RMS is not the identity).
    None,
    Rms(RmsNorm),
}

/// Unweighted per-head normalization of values before they are stored.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValueNorm {
    None,
    RmsUnweighted(UnweightedRms),
}

/// Where a layer's keys and values come from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum KeyValue {
    /// The layer projects and stores its own rows in `domain`.
    Owned {
        /// `[kv_heads·W, hidden]`.
        key: WeightDescriptor,
        value: ValueSource,
        key_norm: HeadNorm,
        value_norm: ValueNorm,
        domain: HistoryDomain,
    },
    /// The layer stores nothing and reads the rows of an earlier attention
    /// sublayer with equal `kv_heads`, `W` and rotary.
    Shared { source: SublayerIndex },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValueSource {
    /// `[kv_heads·W, hidden]`.
    Projected(WeightDescriptor),
    /// Values are the raw key projection, before key norm and rotary.
    Key,
}

/// The set of token rows a layer's history keeps (plan §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HistoryDomain {
    /// One row per token for the whole context.
    Token,
    /// One row per token, keeping only the last `tokens` rows.
    Window { tokens: u64 },
    /// One row per `rate` tokens, committed at block boundaries. Deferred (§9).
    Block { rate: u64 },
}

/// Which history rows a query reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HistoryReads {
    /// Every visible row of the layer's domain.
    Visible,
    /// Rows chosen by a selection index list. Deferred (§9).
    Gathered { rows: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaRowAttention {
    Causal,
    /// Media rows of one span attend to every row of that span.
    Bidirectional,
}

/// Rotary position embedding of queries and keys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Rotary {
    /// No rotary embedding (NoPE).
    None,
    /// Partial rotary embedding with pair axes selected by a repeating
    /// pattern over the input coordinates.
    Interleaved {
        width: u64,
        base: f64,
        sections: Vec<u64>,
        axis_pattern: Vec<u8>,
    },
    /// Pair `p` rotates dimensions `(p, p + P)` by `position·frequency_p`
    /// scaled by `amplitude_p`; dimensions from `2P` pass through.
    Table {
        pairs: Vec<RotaryPair>,
        /// A stored tensor the frequencies were derived from, which the
        /// importer checks at load.
        divisors: Option<RotaryDivisors>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotaryPair {
    pub frequency: f64,
    pub amplitude: f64,
}

/// Stored per-pair frequency divisors (`rope_freqs`, `[P]` F32): pair `p`'s
/// frequency is `bases[p] / d_p`. The family derives the table from headers
/// alone; the importer checks the stored values reproduce it at load and
/// rejects the artifact otherwise: a nonzero frequency must equal
/// `bases[p] / d_p` rounded to f32, and a zero frequency (an unrotated pair)
/// needs `bases[p] / d_p ≤ 1e-20`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotaryDivisors {
    pub weight: WeightDescriptor,
    pub bases: Vec<f64>,
}

/// Deferred (§9): latent multi-query attention over pre-roped latent rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentAttention {
    pub heads: u64,
    pub latent_width: u64,
    pub rotary_width: u64,
}

// --- recurrent mixers --------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentHeadMapping {
    Grouped,
    Tiled,
}

/// Gated DeltaNet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatedDelta {
    pub convolution_width: u64,
    pub key_heads: u64,
    pub value_heads: u64,
    pub width: u64,
    pub head_mapping: RecurrentHeadMapping,
    /// `[channels, hidden]`.
    pub query_key_value: WeightDescriptor,
    /// `[value_heads·W, hidden]`.
    pub gate: WeightDescriptor,
    /// `[value_heads, hidden]`.
    pub alpha: WeightDescriptor,
    pub beta: WeightDescriptor,
    /// `[channels, convolution_width]`.
    pub convolution: WeightDescriptor,
    /// `[value_heads]`.
    pub decay: WeightDescriptor,
    pub time_bias: WeightDescriptor,
    /// Gated normalization of each head's output, `[W]`.
    pub norm: RmsNorm,
    /// `[hidden, value_heads·W]`.
    pub output: WeightDescriptor,
}

impl GatedDelta {
    /// Convolved channels: query and key heads, then value heads.
    pub fn channels(&self) -> Result<u64, DefinitionError> {
        self.key_heads
            .checked_mul(2)
            .and_then(|count| count.checked_add(self.value_heads))
            .and_then(|count| count.checked_mul(self.width))
            .ok_or_else(|| DefinitionError::new("gated delta dimensions overflow"))
    }
}

/// Gated short convolution (LFM2): `y = C ⊙ conv(B ⊙ X)` with a depthwise
/// causal convolution and no bias or activation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShortConv {
    pub channels: u64,
    /// Taps, the current row included.
    pub width: u64,
    /// `B`, `[channels, hidden]`.
    pub input_gate: WeightDescriptor,
    /// `X`, `[channels, hidden]`.
    pub value: WeightDescriptor,
    /// `C`, `[channels, hidden]`.
    pub output_gate: WeightDescriptor,
    /// `[channels, width]`.
    pub convolution: WeightDescriptor,
    /// `[hidden, channels]`.
    pub output: WeightDescriptor,
}

/// Mamba-2 selective state space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateSpace {
    pub heads: u64,
    pub head_width: u64,
    pub state: u64,
    pub groups: u64,
    pub convolution_width: u64,
    /// The input projection, `[inner + channels + heads, hidden]`, rows
    /// `z (inner) | x (inner) | B (groups·state) | C (groups·state) | dt
    /// (heads)`: the gate, the convolved channels and the time step. Head
    /// `h`'s rows are `h·head_width + p`, group `g`'s `g·state + n`. The
    /// state-space kernels consume the projected row whole.
    pub projection: WeightDescriptor,
    /// `[channels, convolution_width]`.
    pub convolution: WeightDescriptor,
    /// `[channels]`.
    pub convolution_bias: WeightDescriptor,
    /// `[heads]`.
    pub time_bias: WeightDescriptor,
    pub decay: WeightDescriptor,
    pub skip: WeightDescriptor,
    /// Normalization after the SiLU(z) gate, over groups of `norm_group`
    /// contiguous channels, `[heads·head_width]`.
    pub norm: RmsNorm,
    pub norm_group: u64,
    /// `[hidden, heads·head_width]`.
    pub output: WeightDescriptor,
}

impl StateSpace {
    pub fn inner(&self) -> Result<u64, DefinitionError> {
        checked_product(&[self.heads, self.head_width])
    }

    /// Convolved channels: `x`, then `B` and `C` of every group.
    pub fn channels(&self) -> Result<u64, DefinitionError> {
        checked_product(&[2, self.groups, self.state])?
            .checked_add(self.inner()?)
            .ok_or_else(|| DefinitionError::new("state space dimensions overflow"))
    }

    /// Rows of the input projection: `z`, the convolved channels and `dt`.
    pub fn projection_rows(&self) -> Result<u64, DefinitionError> {
        self.inner()?
            .checked_add(self.channels()?)
            .and_then(|rows| rows.checked_add(self.heads))
            .ok_or_else(|| DefinitionError::new("state space dimensions overflow"))
    }
}

// --- feed-forward --------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Silu,
    GeluTanh,
    ReluSquared,
}

/// The expanding half of a feed-forward. A stored fused gate‖up tensor is
/// split into `gate` and `up` by import row ranges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeedForwardUp {
    /// `act(gate·x) ⊙ (up·x)`.
    Gated {
        activation: ActivationFunction,
        gate: WeightDescriptor,
        up: WeightDescriptor,
    },
    /// `act(up·x)`.
    Plain {
        activation: ActivationFunction,
        up: WeightDescriptor,
    },
}

impl FeedForwardUp {
    pub fn activation(&self) -> ActivationFunction {
        match self {
            Self::Gated { activation, .. } | Self::Plain { activation, .. } => *activation,
        }
    }

    pub fn up(&self) -> &WeightDescriptor {
        match self {
            Self::Gated { up, .. } | Self::Plain { up, .. } => up,
        }
    }

    pub fn gate(&self) -> Option<&WeightDescriptor> {
        match self {
            Self::Gated { gate, .. } => Some(gate),
            Self::Plain { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DenseFfn {
    pub intermediate: u64,
    /// `[intermediate, hidden]` each.
    pub up: FeedForwardUp,
    /// `[hidden, intermediate]`.
    pub down: WeightDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreFunction {
    Softmax,
    Sigmoid,
    /// Deferred (§9).
    SqrtSoftplus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExpertSelection {
    /// The `selected` highest scores, ranked on `score + bias` when a
    /// selection-only bias (`[experts]`) is present.
    TopK { bias: Option<WeightDescriptor> },
    /// Experts read from the entry hash table. Deferred (§9).
    Hash,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RouterInput {
    /// The routed operator's own normalized input.
    Operator,
    /// Its own normalization of the sublayer's raw residual input, `[hidden]`.
    Residual(RmsNorm),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Router {
    /// `[experts, hidden]`.
    pub weight: WeightDescriptor,
    pub input: RouterInput,
    pub score: ScoreFunction,
    pub selection: ExpertSelection,
    /// How the selected (unbiased) scores become combine weights, as the
    /// model definition states it.
    pub normalization: RouteNormalization,
    /// Multiplier on the final combine weights.
    pub scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RouteNormalization {
    /// The selected scores as they are.
    None,
    /// Divided by their sum.
    Sum,
    /// Divided by `sum + epsilon`.
    SumPlusEpsilon(f64),
    /// Divided by `max(sum, epsilon)`.
    ClampedSum(f64),
}

/// Experts that act in a narrower latent width between a down and an up
/// projection around the routed experts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentExperts {
    pub width: u64,
    /// `[width, hidden]`.
    pub down: WeightDescriptor,
    /// `[hidden, width]`.
    pub up: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SharedExpertGate {
    /// Plain sum with the routed output.
    None,
    /// `sigmoid(w·x)` coefficient, `[hidden]`.
    Sigmoid(WeightDescriptor),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SharedExpert {
    pub intermediate: u64,
    pub up: FeedForwardUp,
    pub down: WeightDescriptor,
    pub gate: SharedExpertGate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoutedFfn {
    pub experts: u64,
    pub selected: u64,
    pub intermediate: u64,
    pub router: Router,
    /// `[experts, intermediate, expert hidden]` each, where the expert
    /// hidden width is the latent width when latent, else `hidden`.
    pub expert_up: FeedForwardUp,
    /// `[experts, expert hidden, intermediate]`.
    pub expert_down: WeightDescriptor,
    /// Per-expert output multiplier, `[experts]`.
    pub expert_scale: Option<WeightDescriptor>,
    pub latent: Option<LatentExperts>,
    pub shared: Option<SharedExpert>,
}

impl RoutedFfn {
    pub fn expert_hidden(&self, hidden: u64) -> u64 {
        self.latent.as_ref().map_or(hidden, |latent| latent.width)
    }
}

/// One layer's per-layer input sublayer (Gemma PLE):
/// `projection · (act(gate · x) ⊙ ple[layer])`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerLayerInput {
    /// Slice of the entry's per-layer table.
    pub layer: u64,
    pub width: u64,
    pub activation: ActivationFunction,
    /// `[width, hidden]`.
    pub gate: WeightDescriptor,
    /// `[hidden, width]`.
    pub projection: WeightDescriptor,
}

/// Forms a model may use that no engine path implements yet (plan §9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeferredForm {
    HyperConnections,
    HyperConnectionMix,
    HyperConnectionHead,
    LatentAttention,
    BlockHistory,
    GatheredHistory,
    SqrtSoftplusScore,
    HashRouting,
}

impl DeferredForm {
    pub fn name(self) -> &'static str {
        match self {
            Self::HyperConnections => "hyper-connection residual streams",
            Self::HyperConnectionMix => "hyper-connection input mixing",
            Self::HyperConnectionHead => "hyper-connection readout head",
            Self::LatentAttention => "latent attention",
            Self::BlockHistory => "block-rate attention history",
            Self::GatheredHistory => "gathered attention history",
            Self::SqrtSoftplusScore => "sqrt-softplus expert scores",
            Self::HashRouting => "hash expert routing",
        }
    }
}

/// The text decoder: entry, blocks and exit over one residual form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decoder {
    pub activation_dtype: crate::ActivationDType,
    pub hidden: u64,
    pub vocabulary: u64,
    pub context_limit: u64,
    pub residual: ResidualForm,
    pub entry: EntryForm,
    pub blocks: Vec<Block>,
    pub exit: ExitForm,
}

impl Decoder {
    /// Every sublayer with its structural address, in execution order.
    pub fn sublayers(&self) -> impl Iterator<Item = (SublayerIndex, &Sublayer)> {
        indexed_sublayers(&self.blocks)
    }

    /// The operator the attention sublayer `index` names, if it is one.
    pub fn attention(&self, index: SublayerIndex) -> Option<&Attention> {
        sublayer(&self.blocks, index).and_then(|sublayer| match &sublayer.op {
            Operator::Attention(attention) => Some(attention.as_ref()),
            _ => None,
        })
    }

    pub fn validate(&self, coordinate_axes: u8) -> Result<(), DefinitionError> {
        if [self.hidden, self.vocabulary, self.context_limit].contains(&0) {
            return Err(DefinitionError::new("invalid decoder dimensions"));
        }
        if self.blocks.is_empty() {
            return Err(DefinitionError::new("decoder has no blocks"));
        }
        if let ResidualForm::HyperConnections(form) = &self.residual {
            if form.streams < 2 {
                return Err(DefinitionError::new(
                    "hyper-connections need at least two streams",
                ));
            }
        }
        self.validate_entry()?;
        let context = Context {
            hidden: self.hidden,
            coordinate_axes,
            per_layer: self.entry.per_layer.as_ref(),
        };
        validate_blocks(&self.blocks, &context)?;
        validate_exit_norm(&self.exit.norm, self.hidden, "decoder output norm")?;
        expect_shape(
            &self.exit.output,
            &[self.vocabulary, self.hidden],
            "decoder output projection",
        )?;
        if self
            .exit
            .softcap
            .is_some_and(|cap| !cap.is_finite() || cap <= 0.0)
        {
            return Err(DefinitionError::new("invalid logit softcap"));
        }
        Ok(())
    }

    fn validate_entry(&self) -> Result<(), DefinitionError> {
        let entry = &self.entry;
        expect_shape(
            &entry.embedding,
            &[self.vocabulary, self.hidden],
            "token embedding",
        )?;
        if let Some(norm) = &entry.norm {
            positive_epsilon(norm.epsilon)?;
        }
        if let Some(per_layer) = &entry.per_layer {
            let rows = checked_product(&[per_layer.layers, per_layer.width])?;
            if per_layer.width == 0
                || per_layer.layers == 0
                || per_layer.media_row >= self.vocabulary
                || [
                    per_layer.table_scale,
                    per_layer.projection_scale,
                    per_layer.combine_scale,
                ]
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
            {
                return Err(DefinitionError::new("invalid per-layer input table"));
            }
            expect_shape(
                &per_layer.table,
                &[self.vocabulary, rows],
                "per-layer input table",
            )?;
            expect_shape(
                &per_layer.projection,
                &[rows, self.hidden],
                "per-layer input projection",
            )?;
            validate_rms(
                &per_layer.projection_norm,
                per_layer.width,
                "per-layer input norm",
            )?;
        }
        if let Some(table) = &entry.hash_routing {
            if table.table.shape.len() != 2 || table.table.shape[0] != self.vocabulary {
                return Err(DefinitionError::new("invalid hash routing table"));
            }
        }
        Ok(())
    }

    pub(crate) fn deferred_forms(&self, forms: &mut Vec<DeferredForm>) {
        if matches!(self.residual, ResidualForm::HyperConnections(_)) {
            forms.push(DeferredForm::HyperConnections);
        }
        if self.entry.hash_routing.is_some() {
            forms.push(DeferredForm::HashRouting);
        }
        blocks_deferred_forms(&self.blocks, forms);
        if matches!(self.exit.norm, ExitNorm::HyperConnectionHead(_)) {
            forms.push(DeferredForm::HyperConnectionHead);
        }
    }
}

pub(crate) fn indexed_sublayers(
    blocks: &[Block],
) -> impl Iterator<Item = (SublayerIndex, &Sublayer)> {
    blocks.iter().enumerate().flat_map(|(block, value)| {
        value
            .sublayers
            .iter()
            .enumerate()
            .map(move |(sublayer, value)| {
                (
                    SublayerIndex {
                        block: block as u32,
                        sublayer: sublayer as u32,
                    },
                    value,
                )
            })
    })
}

fn sublayer(blocks: &[Block], index: SublayerIndex) -> Option<&Sublayer> {
    blocks
        .get(index.block as usize)
        .and_then(|block| block.sublayers.get(index.sublayer as usize))
}

pub(crate) fn blocks_deferred_forms(blocks: &[Block], forms: &mut Vec<DeferredForm>) {
    for block in blocks {
        for sublayer in &block.sublayers {
            if matches!(sublayer.input, InputNorm::HyperConnectionMix(_)) {
                forms.push(DeferredForm::HyperConnectionMix);
            }
            operator_deferred_forms(&sublayer.op, forms);
        }
    }
}

fn operator_deferred_forms(op: &Operator, forms: &mut Vec<DeferredForm>) {
    match op {
        Operator::Attention(attention) => {
            if matches!(
                attention.key_value,
                KeyValue::Owned {
                    domain: HistoryDomain::Block { .. },
                    ..
                }
            ) {
                forms.push(DeferredForm::BlockHistory);
            }
            if matches!(attention.reads, HistoryReads::Gathered { .. }) {
                forms.push(DeferredForm::GatheredHistory);
            }
        }
        Operator::LatentAttention(_) => forms.push(DeferredForm::LatentAttention),
        Operator::RoutedFfn(routed) => {
            if routed.router.score == ScoreFunction::SqrtSoftplus {
                forms.push(DeferredForm::SqrtSoftplusScore);
            }
            if matches!(routed.router.selection, ExpertSelection::Hash) {
                forms.push(DeferredForm::HashRouting);
            }
        }
        Operator::Parallel(branches) => {
            for branch in branches {
                if matches!(branch.input, InputNorm::HyperConnectionMix(_)) {
                    forms.push(DeferredForm::HyperConnectionMix);
                }
                operator_deferred_forms(&branch.op, forms);
            }
        }
        Operator::GatedDelta(_)
        | Operator::ShortConv(_)
        | Operator::StateSpace(_)
        | Operator::DenseFfn(_)
        | Operator::PerLayerInput(_) => {}
    }
}

/// What operator validation needs to know about its decoder.
pub(crate) struct Context<'a> {
    pub hidden: u64,
    pub coordinate_axes: u8,
    pub per_layer: Option<&'a PerLayerEntry>,
}

pub(crate) fn validate_blocks(blocks: &[Block], context: &Context) -> Result<(), DefinitionError> {
    for block in blocks {
        if block.sublayers.is_empty() {
            return Err(DefinitionError::new("block has no sublayers"));
        }
    }
    for (index, sublayer) in indexed_sublayers(blocks) {
        validate_input(&sublayer.input, context.hidden, "sublayer input norm")?;
        validate_operator(&sublayer.op, context)?;
        match &sublayer.output {
            OutputForm::Residual => {}
            OutputForm::PostNorm(norm) => validate_rms(norm, context.hidden, "sublayer post norm")?,
            OutputForm::ScaledPostNorm { norm, layer_scale } => {
                validate_rms(norm, context.hidden, "sublayer post norm")?;
                expect_shape(layer_scale, &[1], "sublayer layer scale")?;
            }
        }
        if let Operator::Attention(attention) = &sublayer.op {
            if let KeyValue::Shared { source } = &attention.key_value {
                let owner = (*source < index)
                    .then(|| self::sublayer(blocks, *source))
                    .flatten()
                    .and_then(|owner| match &owner.op {
                        Operator::Attention(owner) => Some(owner),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        DefinitionError::new(
                            "shared attention history names no earlier attention sublayer",
                        )
                    })?;
                if !matches!(owner.key_value, KeyValue::Owned { .. })
                    || owner.kv_heads != attention.kv_heads
                    || owner.width != attention.width
                    || owner.rotary != attention.rotary
                {
                    return Err(DefinitionError::new(
                        "shared attention history disagrees with its source layer",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_input(input: &InputNorm, hidden: u64, role: &str) -> Result<(), DefinitionError> {
    match input {
        InputNorm::Rms(norm) => validate_rms(norm, hidden, role),
        InputNorm::RmsUnweighted(norm) => positive_epsilon(norm.epsilon),
        InputNorm::HyperConnectionMix(_) | InputNorm::None => Ok(()),
    }
}

fn validate_operator(op: &Operator, context: &Context) -> Result<(), DefinitionError> {
    let hidden = context.hidden;
    match op {
        Operator::Attention(attention) => validate_attention(attention, context),
        Operator::LatentAttention(latent) => {
            if [latent.heads, latent.latent_width].contains(&0)
                || latent.rotary_width > latent.latent_width
            {
                return Err(DefinitionError::new("invalid latent attention geometry"));
            }
            Ok(())
        }
        Operator::GatedDelta(delta) => validate_gated_delta(delta, hidden),
        Operator::ShortConv(conv) => {
            if conv.channels == 0 || conv.width < 2 {
                return Err(DefinitionError::new("invalid short convolution geometry"));
            }
            for (weight, role) in [
                (&conv.input_gate, "short convolution input gate"),
                (&conv.value, "short convolution value"),
                (&conv.output_gate, "short convolution output gate"),
            ] {
                expect_shape(weight, &[conv.channels, hidden], role)?;
            }
            expect_shape(
                &conv.convolution,
                &[conv.channels, conv.width],
                "short convolution taps",
            )?;
            expect_shape(
                &conv.output,
                &[hidden, conv.channels],
                "short convolution output",
            )
        }
        Operator::StateSpace(space) => validate_state_space(space, hidden),
        Operator::DenseFfn(dense) => {
            if dense.intermediate == 0 {
                return Err(DefinitionError::new("invalid dense feed-forward geometry"));
            }
            validate_up(&dense.up, &[dense.intermediate, hidden], "dense feed-forward")?;
            expect_shape(
                &dense.down,
                &[hidden, dense.intermediate],
                "dense feed-forward down projection",
            )
        }
        Operator::RoutedFfn(routed) => validate_routed(routed, hidden),
        Operator::PerLayerInput(input) => {
            let per_layer = context.per_layer.ok_or_else(|| {
                DefinitionError::new("per-layer input sublayer without a per-layer entry table")
            })?;
            if input.width != per_layer.width || input.layer >= per_layer.layers {
                return Err(DefinitionError::new(
                    "per-layer input disagrees with its entry table",
                ));
            }
            expect_shape(&input.gate, &[input.width, hidden], "per-layer input gate")?;
            expect_shape(
                &input.projection,
                &[hidden, input.width],
                "per-layer input projection",
            )
        }
        Operator::Parallel(branches) => {
            if branches.len() < 2 {
                return Err(DefinitionError::new(
                    "parallel sublayer needs at least two branches",
                ));
            }
            for branch in branches {
                if matches!(branch.op, Operator::Parallel(_)) {
                    return Err(DefinitionError::new("parallel branches cannot nest"));
                }
                validate_input(&branch.input, hidden, "branch input norm")?;
                validate_operator(&branch.op, context)?;
                if let BranchOutput::Norm(norm) = &branch.output {
                    validate_rms(norm, hidden, "branch output norm")?;
                }
            }
            Ok(())
        }
    }
}

fn validate_attention(attention: &Attention, context: &Context) -> Result<(), DefinitionError> {
    let hidden = context.hidden;
    let width = attention.width;
    if attention.heads == 0
        || attention.kv_heads == 0
        || width == 0
        || !attention.heads.is_multiple_of(attention.kv_heads)
        || !attention.scale.is_finite()
        || attention.scale <= 0.0
    {
        return Err(DefinitionError::new("invalid attention geometry"));
    }
    let head_rows = checked_product(&[attention.heads, width])?;
    let query_rows = match attention.gate {
        AttentionGate::Interleaved { .. } => checked_product(&[2, head_rows])?,
        _ => head_rows,
    };
    expect_shape(&attention.query, &[query_rows, hidden], "attention query")?;
    if let AttentionGate::Separate {
        weight,
        granularity,
        ..
    } = &attention.gate
    {
        let rows = match granularity {
            GateGranularity::Element => head_rows,
            GateGranularity::Head => attention.heads,
        };
        expect_shape(weight, &[rows, hidden], "attention gate")?;
    }
    validate_head_norm(&attention.query_norm, width, "attention query norm")?;
    if let KeyValue::Owned {
        key,
        value,
        key_norm,
        value_norm,
        domain,
    } = &attention.key_value
    {
        let rows = checked_product(&[attention.kv_heads, width])?;
        expect_shape(key, &[rows, hidden], "attention key")?;
        if let ValueSource::Projected(value) = value {
            expect_shape(value, &[rows, hidden], "attention value")?;
        }
        validate_head_norm(key_norm, width, "attention key norm")?;
        if let ValueNorm::RmsUnweighted(norm) = value_norm {
            positive_epsilon(norm.epsilon)?;
        }
        if matches!(
            domain,
            HistoryDomain::Window { tokens: 0 } | HistoryDomain::Block { rate: 0 }
        ) {
            return Err(DefinitionError::new("invalid attention history domain"));
        }
    }
    match &attention.rotary {
        Rotary::None => {}
        Rotary::Interleaved {
            width: rotary,
            base,
            sections,
            axis_pattern,
        } => {
            let section_width = sections
                .iter()
                .try_fold(0u64, |sum, section| sum.checked_add(*section))
                .and_then(|sum| sum.checked_mul(2));
            if *rotary == 0
                || !rotary.is_multiple_of(2)
                || *rotary > width
                || !base.is_finite()
                || *base <= 0.0
                || section_width != Some(*rotary)
                || axis_pattern.is_empty()
            {
                return Err(DefinitionError::new("invalid rotary geometry"));
            }
            if axis_pattern
                .iter()
                .any(|axis| *axis >= context.coordinate_axes)
            {
                return Err(DefinitionError::new(
                    "rotary axis exceeds the declared input coordinates",
                ));
            }
        }
        Rotary::Table { pairs, divisors } => {
            let rotated = (pairs.len() as u64).checked_mul(2);
            if pairs.is_empty()
                || rotated.is_none_or(|rotated| rotated > width)
                || pairs.iter().any(|pair| {
                    !pair.frequency.is_finite()
                        || pair.frequency < 0.0
                        || !pair.amplitude.is_finite()
                        || pair.amplitude <= 0.0
                })
            {
                return Err(DefinitionError::new("invalid rotary table"));
            }
            if let Some(divisors) = divisors {
                expect_shape(
                    &divisors.weight,
                    &[pairs.len() as u64],
                    "rotary frequency divisors",
                )?;
                if divisors.bases.len() != pairs.len()
                    || divisors
                        .bases
                        .iter()
                        .any(|base| !base.is_finite() || *base <= 0.0)
                {
                    return Err(DefinitionError::new("invalid rotary divisor bases"));
                }
            }
        }
    }
    if let HistoryReads::Gathered { rows: 0 } = attention.reads {
        return Err(DefinitionError::new("invalid gathered history bound"));
    }
    expect_shape(
        &attention.output,
        &[hidden, head_rows],
        "attention output",
    )
}

fn validate_gated_delta(delta: &GatedDelta, hidden: u64) -> Result<(), DefinitionError> {
    if delta.convolution_width < 2
        || delta.key_heads == 0
        || delta.value_heads == 0
        || delta.width == 0
        || !delta.value_heads.is_multiple_of(delta.key_heads)
    {
        return Err(DefinitionError::new("invalid gated delta geometry"));
    }
    let channels = delta.channels()?;
    let inner = checked_product(&[delta.value_heads, delta.width])?;
    expect_shape(
        &delta.query_key_value,
        &[channels, hidden],
        "gated delta query/key/value projection",
    )?;
    expect_shape(&delta.gate, &[inner, hidden], "gated delta gate")?;
    expect_shape(&delta.alpha, &[delta.value_heads, hidden], "gated delta alpha")?;
    expect_shape(&delta.beta, &[delta.value_heads, hidden], "gated delta beta")?;
    expect_shape(
        &delta.convolution,
        &[channels, delta.convolution_width],
        "gated delta convolution",
    )?;
    expect_shape(&delta.decay, &[delta.value_heads], "gated delta decay")?;
    expect_shape(
        &delta.time_bias,
        &[delta.value_heads],
        "gated delta time bias",
    )?;
    validate_rms(&delta.norm, delta.width, "gated delta norm")?;
    expect_shape(&delta.output, &[hidden, inner], "gated delta output")
}

fn validate_state_space(space: &StateSpace, hidden: u64) -> Result<(), DefinitionError> {
    if [
        space.heads,
        space.head_width,
        space.state,
        space.groups,
        space.norm_group,
    ]
    .contains(&0)
        || space.convolution_width < 2
        || !space.heads.is_multiple_of(space.groups)
    {
        return Err(DefinitionError::new("invalid state space geometry"));
    }
    let inner = space.inner()?;
    let channels = space.channels()?;
    if !inner.is_multiple_of(space.norm_group) {
        return Err(DefinitionError::new("invalid state space norm group"));
    }
    expect_shape(
        &space.projection,
        &[space.projection_rows()?, hidden],
        "state space input projection",
    )?;
    expect_shape(
        &space.convolution,
        &[channels, space.convolution_width],
        "state space convolution",
    )?;
    expect_shape(
        &space.convolution_bias,
        &[channels],
        "state space convolution bias",
    )?;
    for (weight, role) in [
        (&space.time_bias, "state space time bias"),
        (&space.decay, "state space decay"),
        (&space.skip, "state space skip"),
    ] {
        expect_shape(weight, &[space.heads], role)?;
    }
    validate_rms(&space.norm, inner, "state space norm")?;
    expect_shape(&space.output, &[hidden, inner], "state space output")
}

fn validate_routed(routed: &RoutedFfn, hidden: u64) -> Result<(), DefinitionError> {
    if [routed.experts, routed.selected, routed.intermediate].contains(&0)
        || routed.selected > routed.experts
        || !routed.router.scale.is_finite()
        || routed.router.scale <= 0.0
    {
        return Err(DefinitionError::new("invalid expert geometry"));
    }
    let router = &routed.router;
    expect_shape(&router.weight, &[routed.experts, hidden], "expert router")?;
    if let RouterInput::Residual(norm) = &router.input {
        validate_rms(norm, hidden, "expert router input norm")?;
    }
    if let ExpertSelection::TopK { bias: Some(bias) } = &router.selection {
        expect_shape(bias, &[routed.experts], "expert selection bias")?;
    }
    let expert_hidden = routed.expert_hidden(hidden);
    validate_up(
        &routed.expert_up,
        &[routed.experts, routed.intermediate, expert_hidden],
        "routed expert",
    )?;
    expect_shape(
        &routed.expert_down,
        &[routed.experts, expert_hidden, routed.intermediate],
        "routed expert down projection",
    )?;
    if let Some(scale) = &routed.expert_scale {
        expect_shape(scale, &[routed.experts], "routed expert scale")?;
    }
    if let Some(latent) = &routed.latent {
        if latent.width == 0 {
            return Err(DefinitionError::new("invalid latent expert width"));
        }
        expect_shape(&latent.down, &[latent.width, hidden], "latent expert down")?;
        expect_shape(&latent.up, &[hidden, latent.width], "latent expert up")?;
    }
    if let Some(shared) = &routed.shared {
        if shared.intermediate == 0 {
            return Err(DefinitionError::new("invalid shared expert geometry"));
        }
        validate_up(
            &shared.up,
            &[shared.intermediate, hidden],
            "shared expert",
        )?;
        expect_shape(
            &shared.down,
            &[hidden, shared.intermediate],
            "shared expert down projection",
        )?;
        if let SharedExpertGate::Sigmoid(gate) = &shared.gate {
            expect_shape(gate, &[hidden], "shared expert gate")?;
        }
    }
    Ok(())
}

fn validate_up(up: &FeedForwardUp, shape: &[u64], role: &str) -> Result<(), DefinitionError> {
    if let Some(gate) = up.gate() {
        expect_shape(gate, shape, &format!("{role} gate projection"))?;
    }
    expect_shape(up.up(), shape, &format!("{role} up projection"))
}

fn validate_head_norm(norm: &HeadNorm, width: u64, role: &str) -> Result<(), DefinitionError> {
    match norm {
        HeadNorm::None => Ok(()),
        HeadNorm::Rms(norm) => validate_rms(norm, width, role),
    }
}

pub(crate) fn validate_rms(norm: &RmsNorm, width: u64, role: &str) -> Result<(), DefinitionError> {
    expect_shape(&norm.weight, &[width], role)?;
    positive_epsilon(norm.epsilon)
}

pub(crate) fn validate_exit_norm(
    norm: &ExitNorm,
    hidden: u64,
    role: &str,
) -> Result<(), DefinitionError> {
    match norm {
        ExitNorm::Rms(norm) => validate_rms(norm, hidden, role),
        ExitNorm::Layer(norm) => {
            expect_shape(&norm.weight, &[hidden], role)?;
            positive_epsilon(norm.epsilon)
        }
        ExitNorm::HyperConnectionHead(_) => Ok(()),
    }
}

fn positive_epsilon(epsilon: f64) -> Result<(), DefinitionError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(DefinitionError::new("invalid normalization epsilon"));
    }
    Ok(())
}
