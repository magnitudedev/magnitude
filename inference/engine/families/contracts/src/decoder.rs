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
//! hyper-connection mixing inside parallel branches, latent attention, and an
//! n-gram input outside the decoder. They are legal descriptions;
//! [`ModelDefinition::deferred_forms`](crate::ModelDefinition::deferred_forms)
//! names them so an executor rejects them with a typed error.

use crate::{DefinitionError, WeightDescriptor, checked_product, expect_shape};
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
    /// `streams` F32 rows of `hidden` per token, entered as copies of the
    /// entry row. Every sublayer reads them through a
    /// [`InputNorm::HyperConnectionMix`] site and writes back through
    /// [`OutputForm::Scatter`]; the exit reads them through
    /// [`ExitNorm::HyperConnectionHead`].
    HyperConnections(HyperConnections),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperConnections {
    pub streams: u64,
    pub mixing: HyperConnectionMixing,
}

/// How every mix site of a decoder forms its coefficients. A site over the
/// streams `X [S, H]` of a row gives the sublayer the input
/// `Σ_s A[s] ⊙ X[s]` and joins its output `y` back as
/// `X'[d] = Σ_s B[s, d]·X[s] + C[d]·y`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HyperConnectionMixing {
    /// Each site's mix passes through a `rank`-wide bottleneck
    /// ([`LowRankGatedSite`]): `A` a per-element gate over the per-stream
    /// normalized streams, `B = I`, `C` the inject values.
    LowRankGated { rank: u64 },
    /// Each site projects the flattened streams once ([`SinkhornSite`]),
    /// in a single pass: `A` the per-stream scalars `pre` of the *previous*
    /// site (a `[S]` F32 carry per row from one site to the next, across
    /// block boundaries; the one-hot `[1, 0, .., 0]` before the first
    /// site), `B` this site's doubly stochastic `comb` after `iterations`
    /// normalizations, `C` its `post` values. `epsilon` is added to `pre`,
    /// to `comb`'s softmax and to every normalization's denominator;
    /// `norm_epsilon` is the projection's RMS statistic's.
    Sinkhorn {
        iterations: u64,
        epsilon: f64,
        norm_epsilon: f64,
    },
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
    /// Collapses hyper-connection streams before readout.
    HyperConnectionHead(HyperConnectionExit),
}

/// The decoder's exit over hyper-connection streams: the streams mixed by
/// `site`'s input coefficients, then normalized by `norm` when present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperConnectionExit {
    pub site: ExitSite,
    pub norm: Option<RmsNorm>,
}

/// Whose input coefficients the exit mixes the streams with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExitSite {
    /// A low-rank site of its own, without inject weights.
    Own(LowRankGatedSite),
    /// The last sublayer site's `pre`, under Sinkhorn mixing.
    PreviousSite,
}

/// One hyper-connection mix site's weights, of the decoder's
/// [`HyperConnectionMixing`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HyperConnectionWeights {
    LowRankGated(LowRankGatedSite),
    Sinkhorn(SinkhornSite),
}

/// One low-rank gated hyper-connection site over `S` streams of `H`, in F32:
///
/// ```text
/// xn[s] = RMS(X[s]) · norm[s]                       (per stream)
/// g     = sigmoid(up · silu(down · xn / S))          ([S, H])
/// y     = mean_s(g[s] ⊙ xn[s])                       (the sublayer input)
/// i     = 2 · sigmoid(inject · xn / S)               ([S], when present)
/// ```
///
/// `y` is the operator's input as it is, not normalized again. A sublayer's
/// [`OutputForm::Scatter`] adds `out · i[s]` to each stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowRankGatedSite {
    /// `[S·H]`, stream-major.
    pub norm: RmsNorm,
    /// `[rank, S·H]`.
    pub down: WeightDescriptor,
    /// `[S·H, rank]`.
    pub up: WeightDescriptor,
    /// `[S, S·H]`; absent at the exit site.
    pub inject: Option<WeightDescriptor>,
}

/// One Sinkhorn hyper-connection site over `S` streams of `H`, in F32, with
/// `x` a row's flattened streams `[S·H]` and the mixing's epsilons and
/// iterations `n`:
///
/// ```text
/// m    = (function · x) · rsqrt(mean(x²) + norm_epsilon)    ([(2+S)·S])
/// pre  = sigmoid(m[0:S]·scale[0] + base[0:S]) + epsilon
/// post = 2·sigmoid(m[S:2S]·scale[1] + base[S:2S])
/// comb = rowsoftmax(m[2S:]·scale[2] + base[2S:]) + epsilon   ([S, S], row s source stream)
/// comb = comb / (colsum(comb) + epsilon), then n - 1 times
///        comb / (rowsum(comb) + epsilon), comb / (colsum(comb) + epsilon)
/// ```
///
/// The sublayer input is `Σ_s pre[s]·X[s]` with the previous site's `pre`,
/// normalized by its input's norm; the
/// sublayer's [`OutputForm::Scatter`] writes `post[d]·y + Σ_s comb[s, d]·X[s]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SinkhornSite {
    /// `[(2+S)·S, S·H]` F32.
    pub function: WeightDescriptor,
    /// `[(2+S)·S]` F32.
    pub base: WeightDescriptor,
    /// `[3]` F32.
    pub scale: WeightDescriptor,
}

/// A sublayer's input through its hyper-connection mix site: the streams
/// mixed by the site's input coefficients, then normalized by `norm` when
/// present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperConnectionInput {
    pub site: HyperConnectionWeights,
    pub norm: Option<RmsNorm>,
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
    /// Mixes hyper-connection streams into the sublayer input.
    HyperConnectionMix(HyperConnectionInput),
    /// The operator reads the raw residual: every stream, under
    /// hyper-connections.
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
    /// `residual + y`; under hyper-connections `y` has a row per stream.
    Residual,
    /// `Σ_s B[s, d]·X[s] + C[d]·y` for every hyper-connection stream `d`,
    /// with `B` and `C` the output coefficients of the sublayer's own mix
    /// site ([`HyperConnectionMixing`]).
    Scatter,
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
    /// Hashed n-gram rows gated into hyper-connection streams. The main
    /// decoder's n-gram input runs (`Decoder::deferred_forms` filters it
    /// out); in a draft head it is still deferred (§9).
    NGramInput(Box<NGramInput>),
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
            Self::NGramInput(_) => "n-gram input",
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
    pub query: QueryProjection,
    pub gate: AttentionGate,
    pub query_norm: HeadNorm,
    /// The history sources one softmax reads, never empty. The first is the
    /// one this layer's fresh rows join: a projected source (its own or a
    /// shared one). Every later source is read as an index list into its
    /// rows: key `o` of the softmax is row `o` of source 0's visible rows,
    /// then the selected rows of each later source in source order.
    pub history: Vec<HistorySource>,
    pub rotary: Rotary,
    /// Multiplier on `q·k` before the softmax, the same for every source.
    pub scale: f64,
    /// Whether media rows of one span see each other.
    pub media_rows: MediaRowAttention,
    /// Per-head attention sinks `[heads]`: query head `h`'s softmax gains
    /// one logit `sinks[h]` (compared with the scaled scores) whose value is
    /// zero, so its output is `Σ e^{s_k}·v_k / (e^{sinks[h]} + Σ e^{s_k})`.
    pub sinks: Option<WeightDescriptor>,
    pub output: OutputProjection,
}

/// The projection of the normed input `x` to the query heads (and, when the
/// gate is interleaved, their gate rows): `Q` rows, `heads·W`, or `2·heads·W`
/// when the gate is interleaved with each head's query rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum QueryProjection {
    /// `[Q, hidden]`.
    Full(WeightDescriptor),
    /// `qr = RMS(down·x)·norm`, then `up·qr`: `down` `[rank, hidden]`, `norm`
    /// `[rank]`, `up` `[Q, rank]`. The normalized `qr` row is an
    /// intermediate a later stage may read.
    LowRank {
        down: WeightDescriptor,
        norm: RmsNorm,
        up: WeightDescriptor,
    },
}

/// The columns of each of a grouped low-rank output's `groups` runs over
/// its `head_rows` (`heads·W`) attended columns
/// ([`OutputProjection::GroupedLowRank`]), or the refusal of a grouping
/// that doesn't split them evenly.
pub fn grouped_output_columns(head_rows: u64, groups: u64) -> Result<u64, DefinitionError> {
    if groups == 0 || !head_rows.is_multiple_of(groups) {
        return Err(DefinitionError::new("invalid grouped attention output"));
    }
    Ok(head_rows / groups)
}

/// The projection of the `heads·W` attended (and gated) row back to the
/// hidden width.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutputProjection {
    /// `[hidden, heads·W]`.
    Full(WeightDescriptor),
    /// The attended row splits into `groups` consecutive runs of
    /// `heads·W / groups` columns; run `g` projects through its own block
    /// `a[g·rank .. (g+1)·rank]` of `a` `[groups·rank, heads·W / groups]`
    /// (block-diagonal), and the `groups·rank` row through `b`
    /// `[hidden, groups·rank]`.
    GroupedLowRank {
        groups: u64,
        rank: u64,
        a: WeightDescriptor,
        b: WeightDescriptor,
    },
}

/// A rotation of each query head's attended row, before the gate and the
/// output projection, by its rotary table at the row's own coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputRotary {
    None,
    /// The inverse rotation (negated angles) of the one applied to queries
    /// and keys, in the table's layout.
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateFunction {
    Sigmoid,
    Softplus,
    Silu,
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
    Interleaved {
        function: GateFunction,
    },
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

/// One source of rows an attention softmax reads, and which of them a query
/// reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistorySource {
    pub rows: HistoryRows,
    pub reads: HistoryReads,
}

impl HistorySource {
    /// Every visible row of `rows`.
    pub fn visible(rows: HistoryRows) -> Self {
        Self {
            rows,
            reads: HistoryReads::Visible,
        }
    }
}

/// Where a source's rows come from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HistoryRows {
    /// The layer produces and stores the rows itself.
    Own(RowProducer),
    /// The layer stores nothing for this source and reads source `part` of
    /// an earlier attention sublayer `source` (a target's, for a head
    /// reading target history), which owns it, with equal row geometry and
    /// key rotary ([`Rotary::of_keys`]).
    Shared { source: SublayerIndex, part: u64 },
}

/// How a source's rows are made from the layer's normed input `x`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RowProducer {
    /// One row of `kv_heads` keys (and values) per token.
    Projected(ProjectedRows),
    /// One row of one `W`-wide head per `rate` tokens, read as both key and
    /// value.
    Pooled(PooledRows),
}

impl RowProducer {
    /// The domain of the rows it produces.
    pub fn domain(&self) -> HistoryDomain {
        match self {
            Self::Projected(rows) => rows.domain.into(),
            Self::Pooled(rows) => HistoryDomain::Block { rate: rows.rate },
        }
    }
}

impl Attention {
    /// The source this layer's fresh rows join: its first.
    pub fn fresh(&self) -> &HistorySource {
        &self.history[0]
    }
}

/// Per-token keys and values: key `rotary(key_norm(key·x))`, then
/// `rounding`; the value by `value`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedRows {
    /// `[kv_heads·W, hidden]`.
    pub key: WeightDescriptor,
    pub value: ValueSource,
    pub key_norm: HeadNorm,
    pub value_norm: ValueNorm,
    pub domain: TokenDomain,
    /// The trained rounding of each stored row, after norm and rotary:
    /// the row attention reads, fresh or from history.
    pub rounding: RowRounding,
}

/// The token-rate domains a projected source keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenDomain {
    /// One row per token for the whole context.
    Token,
    /// One row per token, keeping only the last `tokens` rows.
    Window { tokens: u64 },
}

impl From<TokenDomain> for HistoryDomain {
    fn from(domain: TokenDomain) -> Self {
        match domain {
            TokenDomain::Token => Self::Token,
            TokenDomain::Window { tokens } => Self::Window { tokens },
        }
    }
}

/// One row per complete group of `rate` tokens, `[t·rate, (t+1)·rate)`:
///
/// ```text
/// u_i     = projection·x_i                       (F32, per token i of the group)
/// pooled  = pool(u)                              (per channel)
/// latent  = RMS(pooled)·norm
/// row     = rounding(rotary(latent, coordinates(first token of the group)))
/// ```
///
/// The rotary is the reading attention's key rotary in its layout over `W`.
/// A group's row exists once its last token has been seen, so a query at `t`
/// reads rows `b < (t + 1) / rate`. The tokens of the incomplete group are
/// producer state: their F32 projections (and gate scores) in a
/// `Window(rate − 1)` domain, which rollback, fork and checkpoints treat as
/// any window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PooledRows {
    pub rate: u64,
    /// `[W, hidden]`.
    pub projection: WeightDescriptor,
    pub pool: Pool,
    /// `[W]`.
    pub norm: RmsNorm,
    pub rounding: RowRounding,
}

/// How a group's per-token projections become one row, per channel `d`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Pool {
    /// `rate == 1`: the token's projection itself.
    Identity,
    /// `rate >= 2`: `Σ_i u_i[d] / rate`.
    Mean,
    /// `rate >= 2`: `Σ_i softmax_i(g[d])·u_i[d]`, with `g_i = gate·x_i`
    /// (`gate` `[W, hidden]`, F32) and the softmax over the group's tokens.
    SoftmaxGated { gate: WeightDescriptor },
}

/// A trained rounding of a row's values, part of the model's numerics: the
/// row is computed, then rounded, and every reader sees the rounded values.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RowRounding {
    None,
    Block(BlockRounding),
}

/// Each run of `block` consecutive values `x` is rounded to
/// `element(clamp(x / s, ±m)) · s`, `element` rounding to the nearest code,
/// ties to even, `m` the element's largest finite value, with one scale per
/// run, computed in F32 from `a = max(max|x|, amax_floor)`:
/// - [`BlockScale::Ue8m0`]: `s = 2^ceil(log2(a · f32(1/m)))`;
/// - [`BlockScale::E4M3`]: `s = e4m3(a / m)`, to the nearest code.
///
/// `amax_floor` is the model's own (it keeps an all-zero run's scale
/// finite and nonzero: validation refuses a floor whose scale rounds to
/// zero). Every value it yields is exact in BF16.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockRounding {
    pub element: BlockElement,
    pub block: u64,
    pub scale: BlockScale,
    pub amax_floor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockElement {
    /// FP8 E4M3 (`m = 448`).
    E4M3,
    /// FP4 E2M1 (`m = 6`).
    E2M1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockScale {
    /// A power of two.
    Ue8m0,
    /// An FP8 E4M3 value.
    E4M3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValueSource {
    /// `[kv_heads·W, hidden]`.
    Projected(WeightDescriptor),
    /// Values are the raw key projection, before key norm and rotary.
    RawKey,
    /// Values are the stored key row, after key norm and rotary: the history
    /// holds one plane, read as both. Takes no value norm.
    StoredKey,
}

/// The set of token rows a layer's history keeps (plan §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HistoryDomain {
    /// One row per token for the whole context.
    Token,
    /// One row per token, keeping only the last `tokens` rows.
    Window { tokens: u64 },
    /// One row per `rate` tokens, committed at block boundaries: a pooled
    /// source's rows, and the keys of an indexer over them.
    Block { rate: u64 },
}

/// Which rows of a source a query reads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HistoryReads {
    /// Every visible row of the source.
    Visible,
    /// The rows a selection picks.
    Selected(Selection),
}

/// Where a selected read's picks come from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Selection {
    /// The layer's own indexer.
    Own(Box<Indexer>),
    /// The picks of source `part`'s own indexer in the earlier attention
    /// sublayer `source`, over the same rows (the latest index source a
    /// Reuse layer follows): read by reference, computed once.
    Reuse { source: SublayerIndex, part: u64 },
}

/// Scores the units of a source's rows and selects the best, in F32:
///
/// ```text
/// q[h]    = rounding(rotary(norm(query·input)[h], coordinates(t)))
/// K[u]    = the keys' row u                               (one per unit)
/// s[t, u] = scale · Σ_h relu(q[h] · K[u]) · w[h]
/// ```
///
/// with `w[h] = 1` ([`HeadWeights::Uniform`]) or `(weight·x)[h]`. The
/// rotary is the attention's own, in its layout over each `width`.
///
/// A unit is the run of the read source's rows one key covers: key `u`
/// covers rows `[u·n, (u + 1)·n)`, `n` the keys' rate over the source's
/// (a pooled key over a token source: its group of tokens; a key per row
/// of a pooled source: that row). A row at `t` has `c` complete units.
/// While `c <= top` it reads every visible row of the source; past it, the
/// rows of its `top` best-scoring complete units (ties select the lower
/// unit) and the visible rows no complete unit covers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Indexer {
    pub heads: u64,
    pub width: u64,
    pub query: IndexerQuery,
    pub keys: IndexerKeys,
    pub head_weights: HeadWeights,
    /// Multiplier on the summed head scores.
    pub scale: f64,
    pub top: u64,
    pub candidates: Candidates,
}

/// Which complete units of its row an indexer scores: a two-level top-k,
/// whose first level one layer produces and later layers restrict to.
///
/// A pool of `blocks` blocks of `block_units` consecutive units, per row of
/// `c` complete units: block `b` holds the units `[b·block_units,
/// (b + 1)·block_units)` below `c` and scores the largest of their scores;
/// the row's newest block (the one holding unit `c − 1`) scores `+inf`; the
/// pool is the `blocks` best blocks, ties to the lower block. A restricted
/// row scores every unit outside the pool `−inf`, then takes its `top` best
/// units as any row does (ties to the lower unit), so a pool holding fewer
/// than `min(top, c)` units adds the lowest units outside it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Candidates {
    /// Every complete unit.
    All,
    /// Every complete unit, and the indexer's scores make the pool that later
    /// indexers over the same rows restrict to.
    Produce { blocks: u64, block_units: u64 },
    /// The units of the pool produced by source `part`'s indexer in the
    /// earlier attention sublayer `source`, over the same rows.
    Restrict { source: SublayerIndex, part: u64 },
}

/// The indexer's query heads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexerQuery {
    pub input: IndexerInput,
    /// `[heads·width, input width]`.
    pub projection: WeightDescriptor,
    /// Per head, over `width`.
    pub norm: HeadNorm,
    pub rounding: RowRounding,
}

/// The row an indexer's query projects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexerInput {
    /// The layer's normed input `x` (`hidden` wide).
    Hidden,
    /// The attention's normalized low-rank query row `qr`
    /// ([`QueryProjection::LowRank`], `rank` wide).
    QueryRank,
}

/// The rows an indexer scores, one per unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexerKeys {
    /// Its own pooled rows of `x`, one head `width` wide, over a token
    /// source.
    Pooled(PooledRows),
    /// Per row of the read source (a pooled one, which this layer owns):
    /// `rounding(rotary(RMS(projection·latent)·norm, coordinates(first token
    /// of the row)))`, `latent` the source's row before its rotary and
    /// rounding. `projection` `[width, W]`, `norm` `[width]`.
    Latent {
        projection: WeightDescriptor,
        norm: RmsNorm,
        rounding: RowRounding,
    },
    /// The keys of source `part`'s own indexer in the earlier attention
    /// sublayer `source`, which owns the rows this source reads (its
    /// [`IndexerKeys::Latent`] keys): read by reference, computed once. The
    /// layer projects its own query and head weights.
    Shared { source: SublayerIndex, part: u64 },
}

/// The per-head weights of an indexer's summed scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HeadWeights {
    /// Every head weighs 1.
    Uniform,
    /// `w = weight·x`, `weight` `[heads, hidden]`.
    Projected(WeightDescriptor),
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
    /// Pair `p` rotates its two dimensions (by `layout`) by
    /// `position·frequency_p` scaled by `amplitude_p`; the other `W − 2P`
    /// dimensions pass through.
    Table {
        pairs: Vec<RotaryPair>,
        layout: RotaryLayout,
        /// A stored tensor the frequencies were derived from, which the
        /// importer checks at load.
        divisors: Option<RotaryDivisors>,
        output: OutputRotary,
    },
}

impl Rotary {
    /// The rotary as it rotates queries and keys, without what it does to
    /// the attended row: what a layer's stored history rows depend on.
    pub fn of_keys(&self) -> Self {
        match self {
            Self::Table { pairs, layout, divisors, output: _ } => Self::Table {
                pairs: pairs.clone(),
                layout: *layout,
                divisors: divisors.clone(),
                output: OutputRotary::None,
            },
            Self::None | Self::Interleaved { .. } => self.clone(),
        }
    }
}

/// Which dimensions of a `W`-wide head a rotary table's `P` pairs rotate:
/// the rotated `2P` dimensions are the head's first (`Leading`) or last
/// (`Trailing`) `2P`, and within them pair `p` is `(p, p + P)` (`HalfSplit`)
/// or `(2p, 2p + 1)` (`Adjacent`). The layout is the checkpoint's own, kept in
/// the kernels: permuting weight rows at import would move dimensions across
/// a stored row's rounding blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RotaryLayout {
    pub pairs: RotaryPairing,
    pub placement: RotaryPlacement,
}

impl RotaryLayout {
    /// Pairs `(p, p + P)` over the first `2P` dimensions.
    pub const HALF_SPLIT_LEADING: Self = Self {
        pairs: RotaryPairing::HalfSplit,
        placement: RotaryPlacement::Leading,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RotaryPairing {
    HalfSplit,
    Adjacent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RotaryPlacement {
    Leading,
    Trailing,
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
    /// Gated normalization of each head's output, `[W]`:
    /// `RMS(o)·norm ⊙ output_gate(gate·x)`.
    pub norm: RmsNorm,
    pub output_gate: GateFunction,
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

/// A gated expansion's clamp L (DeepSeek V4.1's `swiglu_limit`): positive
/// and finite at F32, the width the kernels clamp at, by construction; a
/// definition carrying any other value does not deserialize. Equal by bits,
/// which is equality for these values.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct GluLimit(f64);

impl GluLimit {
    pub fn new(limit: f64) -> Result<Self, DefinitionError> {
        let word = limit as f32;
        if word > 0.0 && word.is_finite() {
            Ok(Self(limit))
        } else {
            Err(DefinitionError::new(format!("invalid expansion limit {limit}")))
        }
    }

    /// The static `LB` of the kernels' gated expansions (functions.seismic's
    /// `glu`) for an expansion's clamp: the bits of L at F32, or 0, which no
    /// limit takes, for none. A kernel compiles the clamp in only where it is
    /// nonzero.
    pub fn word(limit: Option<Self>) -> u64 {
        limit.map_or(0, |limit| u64::from((limit.0 as f32).to_bits()))
    }
}

impl TryFrom<f64> for GluLimit {
    type Error = DefinitionError;

    fn try_from(limit: f64) -> Result<Self, DefinitionError> {
        Self::new(limit)
    }
}

impl From<GluLimit> for f64 {
    fn from(limit: GluLimit) -> f64 {
        limit.0
    }
}

impl PartialEq for GluLimit {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for GluLimit {}

impl std::hash::Hash for GluLimit {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// The expanding half of a feed-forward. A stored fused gate‖up tensor is
/// split into `gate` and `up` by import row ranges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeedForwardUp {
    /// `act(gate·x) ⊙ (up·x)`, or with a `limit` L,
    /// `act(min(gate·x, L)) ⊙ clamp(up·x, -L, L)` (DeepSeek V4.1's
    /// `swiglu_limit`: the gate above only, up on both sides).
    Gated {
        activation: ActivationFunction,
        limit: Option<GluLimit>,
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

    /// The gated expansion's clamp (`Gated`'s `limit`); an up-only one has
    /// none.
    pub fn limit(&self) -> Option<GluLimit> {
        match self {
            Self::Gated { limit, .. } => *limit,
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
    /// `sqrt(softplus(logit))`, in F32.
    SqrtSoftplus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExpertSelection {
    /// The `selected` highest scores, ranked on `score + bias` when a
    /// selection-only bias (`[experts]`) is present.
    TopK { bias: Option<WeightDescriptor> },
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
    /// The selection-only bias (`[experts]`) media rows rank on in place of
    /// the selection's own (DeepSeek V4.1's `bias_vl`); text rows never
    /// read it.
    pub media_bias: Option<WeightDescriptor>,
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

/// Hashed n-gram rows gated into each of `S` hyper-connection streams of
/// `H` (Qwen4's per-layer input, DeepSeek V4.1's Engram). It reads the raw
/// streams `X` and adds its `[S, H]` output to them, in F32:
///
/// ```text
/// e    = concat_h(table[row_h(t)])                        ([heads·width])
/// k[s] = RMS(key·e)[s];  q[s] = RMS(X[s])
/// g[s] = Σ_c k[s,c]·q[s,c]·w[s,c] / sqrt(H);  g = sign(g)·sqrt(max(|g|, 1e-6))
/// v[s] = sigmoid(g[s]) · (value·e)
/// y    = v                                    (no mixing)
/// y    = v + silu(conv(RMS(v)·norm))          (a convolution)
/// ```
///
/// `w` is the gate's weight (`NGramGate`). `conv` is depthwise and causal
/// over each stream channel:
/// `conv(x)[t] = Σ_j taps[j]·x[t − dilation·(kernel − 1 − j)]`, rows before
/// the sequence reading zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NGramInput {
    pub hash: NGramHash,
    /// Row width.
    pub width: u64,
    /// `[rows, width]`, gathered `hash.heads()` rows per position.
    pub table: WeightDescriptor,
    /// `[S·H, heads·width]`.
    pub key: WeightDescriptor,
    /// `[H, heads·width]`.
    pub value: WeightDescriptor,
    pub gate: NGramGate,
    /// The epsilon of every RMS the input takes.
    pub epsilon: f64,
    /// The convolution of the gated value, when the input mixes it.
    pub mixing: Option<NGramConvolution>,
}

/// The gate's per-column weight `w` (`[S·H]`, applied per stream).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NGramGate {
    /// The key's and the query's RMS norm weights: `w = key·query`.
    Norms {
        key: WeightDescriptor,
        query: WeightDescriptor,
    },
    /// The product itself, stored as one weight.
    Product(WeightDescriptor),
}

/// The convolution an n-gram input mixes its gated value with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NGramConvolution {
    /// `[S·H]`, the RMS norm weight of `v`.
    pub norm: WeightDescriptor,
    /// `[S·H, kernel]`.
    pub taps: WeightDescriptor,
    pub kernel: u64,
    pub dilation: u64,
}

/// The table rows of position `t`. A token hashes as `token_map[token]`, or
/// as itself without a map. With `c[0]` the hashed `token(t)` and `c[d]` its
/// `d`-th predecessor's, a predecessor that is missing or at or before the
/// latest `boundary` token preceding `t` reads as `fill` (a boundary at `t`
/// itself still reads its own predecessors). Order `n` (`2..=context + 1`)
/// hashes `m = XOR_{d < n} c[d]·multipliers[d]`, exact in `u64`; its
/// `heads_per_order` heads `h` read row `offsets[h] + m mod sizes[h]`.
/// Media rows hash as `media` states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NGramHash {
    /// Predecessors read.
    pub context: u64,
    pub heads_per_order: u64,
    /// `[context + 1]`.
    pub multipliers: Vec<u64>,
    /// `[heads]` each, order-major.
    pub sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// `[vocabulary]`: the hashed id of each raw token.
    pub token_map: Option<Vec<u32>>,
    /// The hashed id a missing or cut predecessor reads.
    pub fill: u64,
    /// The raw token that cuts the lookback of the positions after it.
    pub boundary: Option<u64>,
    pub media: NGramMedia,
}

/// How a media row (a vision row in place of its token) hashes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NGramMedia {
    /// As the token that stands in its place.
    Hashed,
    /// As `fill`, with every older predecessor read as `fill` and the row's
    /// gate 0.
    Blocking,
}

impl NGramHash {
    pub fn heads(&self) -> Result<u64, DefinitionError> {
        checked_product(&[self.context, self.heads_per_order])
    }

    /// The ids tokens hash as: the map's, or a decoder `vocabulary`'s.
    pub fn hashed_vocabulary(&self, vocabulary: u64) -> u64 {
        match &self.token_map {
            Some(map) => map.iter().max().map_or(0, |id| u64::from(*id) + 1),
            None => vocabulary,
        }
    }

    /// The table rows of every position of `tokens`, `heads()` per position
    /// in head order. `preceding` is the sequence's tokens before the first,
    /// oldest first: its last `context` tokens, or every token of a sequence
    /// that has fewer.
    pub fn rows(&self, preceding: &[u64], tokens: &[u64]) -> Vec<u64> {
        let context = self.context as usize;
        let heads_per_order = self.heads_per_order as usize;
        let preceding = &preceding[preceding.len().saturating_sub(context)..];
        let history = preceding.iter().chain(tokens).copied().collect::<Vec<_>>();
        let hashed = |token: u64| match &self.token_map {
            Some(map) => u64::from(map[token as usize]),
            None => token,
        };
        let mut rows = Vec::with_capacity(tokens.len() * context * heads_per_order);
        // The latest boundary strictly before the position.
        let mut latest = self
            .boundary
            .and_then(|boundary| preceding.iter().rposition(|token| *token == boundary));
        let mut c = vec![0u64; context + 1];
        for at in preceding.len()..history.len() {
            if at > preceding.len() && Some(history[at - 1]) == self.boundary {
                latest = Some(at - 1);
            }
            c[0] = hashed(history[at]);
            for distance in 1..=context {
                c[distance] = match at.checked_sub(distance) {
                    Some(source) if latest.is_none_or(|cut| source > cut) => {
                        hashed(history[source])
                    }
                    _ => self.fill,
                };
            }
            for order in 2..=context + 1 {
                let mixed = (0..order).fold(0u64, |mixed, distance| {
                    mixed ^ c[distance].wrapping_mul(self.multipliers[distance])
                });
                let first = (order - 2) * heads_per_order;
                rows.extend(
                    (first..first + heads_per_order)
                        .map(|head| self.offsets[head] + mixed % self.sizes[head]),
                );
            }
        }
        rows
    }
}

/// Forms a model may use that no engine path implements yet (plan §9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeferredForm {
    HyperConnectionMix,
    HyperConnectionHead,
    LatentAttention,
    NGramInput,
}

impl DeferredForm {
    pub fn name(self) -> &'static str {
        match self {
            Self::HyperConnectionMix => "hyper-connection input mixing",
            Self::HyperConnectionHead => "hyper-connection readout head",
            Self::LatentAttention => "latent attention",
            Self::NGramInput => "n-gram input",
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

    /// The domain of the source the attention sublayer `index`'s fresh rows
    /// join (its first): its own, or that of the layer it shares.
    pub fn history_domain(&self, index: SublayerIndex) -> Option<HistoryDomain> {
        source_producer(&self.blocks, index, 0).map(RowProducer::domain)
    }

    /// The producer of source `part` of the attention sublayer `index`: its
    /// own, or that of the layer it shares.
    pub fn source_producer(&self, index: SublayerIndex, part: u64) -> Option<&RowProducer> {
        source_producer(&self.blocks, index, part)
    }

    pub fn validate(&self, coordinate_axes: u8) -> Result<(), DefinitionError> {
        if [self.hidden, self.vocabulary, self.context_limit].contains(&0) {
            return Err(DefinitionError::new("invalid decoder dimensions"));
        }
        if self.blocks.is_empty() {
            return Err(DefinitionError::new("decoder has no blocks"));
        }
        if let ResidualForm::HyperConnections(form) = &self.residual {
            let invalid = match form.mixing {
                HyperConnectionMixing::LowRankGated { rank } => rank == 0,
                HyperConnectionMixing::Sinkhorn {
                    iterations,
                    epsilon,
                    norm_epsilon,
                    ..
                } => {
                    iterations == 0
                        || positive_epsilon(epsilon).is_err()
                        || positive_epsilon(norm_epsilon).is_err()
                }
            };
            if form.streams < 2 || invalid {
                return Err(DefinitionError::new("invalid hyper-connection geometry"));
            }
        }
        self.validate_entry()?;
        let context = Context {
            hidden: self.hidden,
            vocabulary: self.vocabulary,
            coordinate_axes,
            residual: &self.residual,
            per_layer: self.entry.per_layer.as_ref(),
            shared_sources: None,
        };
        validate_blocks(&self.blocks, &context)?;
        validate_exit_norm(&self.exit.norm, &context, "decoder output norm")?;
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
        Ok(())
    }

    pub(crate) fn deferred_forms(&self, forms: &mut Vec<DeferredForm>) {
        for block in &self.blocks {
            for sublayer in &block.sublayers {
                let mut operator = Vec::new();
                operator_deferred_forms(&sublayer.op, &mut operator);
                // The decoder's n-gram input runs (`ngram.seismic`).
                forms.extend(
                    operator
                        .into_iter()
                        .filter(|form| *form != DeferredForm::NGramInput),
                );
            }
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

/// The deferred operators of a draft's or head's blocks.
pub(crate) fn blocks_deferred_forms(blocks: &[Block], forms: &mut Vec<DeferredForm>) {
    for block in blocks {
        for sublayer in &block.sublayers {
            operator_deferred_forms(&sublayer.op, forms);
        }
    }
}

fn operator_deferred_forms(op: &Operator, forms: &mut Vec<DeferredForm>) {
    match op {
        Operator::Attention(_) => {}
        Operator::LatentAttention(_) => forms.push(DeferredForm::LatentAttention),
        Operator::NGramInput(_) => forms.push(DeferredForm::NGramInput),
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
        | Operator::RoutedFfn(_)
        | Operator::PerLayerInput(_) => {}
    }
}

/// What operator validation needs to know about its decoder.
pub(crate) struct Context<'a> {
    pub hidden: u64,
    pub vocabulary: u64,
    pub coordinate_axes: u8,
    pub residual: &'a ResidualForm,
    pub per_layer: Option<&'a PerLayerEntry>,
    /// The blocks a `HistoryRows::Shared` source names when they are not the
    /// validated blocks themselves: the target decoder's, for a draft head
    /// reading target history. Every attention source is then Shared.
    pub shared_sources: Option<&'a [Block]>,
}

fn attention_at(blocks: &[Block], index: SublayerIndex) -> Option<&Attention> {
    sublayer(blocks, index).and_then(|sublayer| match &sublayer.op {
        Operator::Attention(attention) => Some(attention.as_ref()),
        _ => None,
    })
}

/// The owner of source `part` of the attention sublayer `index` in
/// `blocks`, with the owner's attention: `(index, part)` itself when it owns
/// the rows, or the owned source a Shared one names.
pub(crate) fn source_owner(
    blocks: &[Block],
    index: SublayerIndex,
    part: u64,
) -> Option<(SublayerIndex, u64, &Attention)> {
    let attention = attention_at(blocks, index)?;
    match &attention.history.get(part as usize)?.rows {
        HistoryRows::Own(_) => Some((index, part, attention)),
        HistoryRows::Shared { source, part } => {
            let owner = attention_at(blocks, *source)?;
            matches!(owner.history.get(*part as usize)?.rows, HistoryRows::Own(_))
                .then_some((*source, *part, owner))
        }
    }
}

/// The producer of source `part` of the attention sublayer `index`.
pub(crate) fn source_producer(
    blocks: &[Block],
    index: SublayerIndex,
    part: u64,
) -> Option<&RowProducer> {
    let (_, part, owner) = source_owner(blocks, index, part)?;
    match &owner.history[part as usize].rows {
        HistoryRows::Own(producer) => Some(producer),
        HistoryRows::Shared { .. } => None,
    }
}

pub(crate) fn validate_blocks(blocks: &[Block], context: &Context) -> Result<(), DefinitionError> {
    for block in blocks {
        if block.sublayers.is_empty() {
            return Err(DefinitionError::new("block has no sublayers"));
        }
    }
    for (index, sublayer) in indexed_sublayers(blocks) {
        validate_residual_use(sublayer, context.residual)?;
        validate_input(&sublayer.input, context, "sublayer input norm")?;
        validate_operator(&sublayer.op, context)?;
        match &sublayer.output {
            OutputForm::Residual | OutputForm::Scatter => {}
            OutputForm::PostNorm(norm) => validate_rms(norm, context.hidden, "sublayer post norm")?,
            OutputForm::ScaledPostNorm { norm, layer_scale } => {
                validate_rms(norm, context.hidden, "sublayer post norm")?;
                expect_shape(layer_scale, &[1], "sublayer layer scale")?;
            }
        }
        if let Operator::Attention(attention) = &sublayer.op {
            validate_shared_sources(blocks, index, attention, context)?;
            validate_output_rotary(blocks, attention, context)?;
        }
    }
    Ok(())
}

/// An inverse output rotary undoes the key rotation the attended values
/// carry: only stored-key values (the first source's, own or shared) are
/// rotated keys. Pooled sources, which follow the first, are always rotated.
fn validate_output_rotary(
    blocks: &[Block],
    attention: &Attention,
    context: &Context,
) -> Result<(), DefinitionError> {
    let Rotary::Table {
        output: OutputRotary::Inverse,
        ..
    } = &attention.rotary
    else {
        return Ok(());
    };
    let producer = match &attention.history[0].rows {
        HistoryRows::Own(producer) => Some(producer),
        HistoryRows::Shared { source, part } => {
            source_producer(context.shared_sources.unwrap_or(blocks), *source, *part)
        }
    };
    match producer {
        Some(RowProducer::Projected(ProjectedRows {
            value: ValueSource::StoredKey,
            ..
        })) => Ok(()),
        _ => Err(DefinitionError::new(
            "an inverse output rotary over values other than the stored keys",
        )),
    }
}

/// A layer's Shared sources and reused selections against the layers they
/// name: an earlier sublayer of `blocks` that owns the rows (or, for a head
/// reading target history, the target's owner of them), with equal row
/// geometry and key rotary.
fn validate_shared_sources(
    blocks: &[Block],
    index: SublayerIndex,
    attention: &Attention,
    context: &Context,
) -> Result<(), DefinitionError> {
    for (part, source) in attention.history.iter().enumerate() {
        let owner = match (&source.rows, context.shared_sources) {
            (HistoryRows::Own(_), None) => None,
            (HistoryRows::Own(_), Some(_)) => {
                return Err(DefinitionError::new(
                    "attention reading target history owns its own",
                ));
            }
            (HistoryRows::Shared { source, part }, None) => Some(
                (*source < index)
                    .then(|| attention_at(blocks, *source))
                    .flatten()
                    .and_then(|owner| {
                        let rows = &owner.history.get(*part as usize)?.rows;
                        matches!(rows, HistoryRows::Own(_)).then_some(owner)
                    })
                    .map(|owner| (owner, *part))
                    .ok_or_else(|| {
                        DefinitionError::new(
                            "shared attention history names no earlier attention sublayer",
                        )
                    })?,
            ),
            (HistoryRows::Shared { source, part }, Some(target)) => Some(
                source_owner(target, *source, *part)
                    .map(|(_, part, owner)| (owner, part))
                    .ok_or_else(|| {
                        DefinitionError::new(
                            "shared attention history names no target attention sublayer",
                        )
                    })?,
            ),
        };
        if let Some((owner, owned)) = owner {
            let agrees = match &owner.history[owned as usize].rows {
                HistoryRows::Own(RowProducer::Projected(_)) => {
                    part == 0 && owner.kv_heads == attention.kv_heads
                }
                HistoryRows::Own(RowProducer::Pooled(_)) => part > 0,
                HistoryRows::Shared { .. } => false,
            };
            if !agrees
                || owner.width != attention.width
                || owner.rotary.of_keys() != attention.rotary.of_keys()
            {
                return Err(DefinitionError::new(
                    "shared attention history disagrees with its source layer",
                ));
            }
        }
        if let HistoryReads::Selected(Selection::Reuse {
            source: reused,
            part: reused_part,
        }) = &source.reads
        {
            // Every layer reading these rows binds one launch-scoped picks
            // table and an own indexer overwrites it, so the reused picks
            // are the latest own indexer's over the rows this source reads.
            let rows = source_owner(blocks, index, part as u64).map(|(a, ap, _)| (a, ap));
            let indexers: Vec<(SublayerIndex, u64)> = indexed_sublayers(blocks)
                .take_while(|(at, _)| *at < index)
                .filter_map(|(at, sublayer)| match &sublayer.op {
                    Operator::Attention(layer) => Some((at, layer)),
                    _ => None,
                })
                .flat_map(|(at, layer)| {
                    layer.history.iter().enumerate().filter_map(move |(at_part, picked)| {
                        matches!(picked.reads, HistoryReads::Selected(Selection::Own(_)))
                            .then_some((at, at_part as u64))
                    })
                })
                .filter(|(at, at_part)| {
                    rows.is_some()
                        && source_owner(blocks, *at, *at_part).map(|(b, bp, _)| (b, bp)) == rows
                })
                .collect();
            let named = (*reused, *reused_part);
            if context.shared_sources.is_some() || !indexers.contains(&named) {
                return Err(DefinitionError::new(
                    "reused selection names no earlier indexer over the same rows",
                ));
            }
            if indexers.last() != Some(&named) {
                return Err(DefinitionError::new(
                    "reused selection names an indexer other than the latest over the same rows",
                ));
            }
        }
        if let HistoryReads::Selected(Selection::Own(indexer)) = &source.reads {
            validate_shared_indexer(blocks, index, part as u64, attention, indexer, context)?;
        }
    }
    Ok(())
}

/// An own indexer's references to earlier indexers over the same rows: the
/// keys it shares (an indexer keeping its own), and the candidate pool it
/// restricts to (one that produces a pool of at least its `top` units).
fn validate_shared_indexer(
    blocks: &[Block],
    index: SublayerIndex,
    part: u64,
    attention: &Attention,
    indexer: &Indexer,
    context: &Context,
) -> Result<(), DefinitionError> {
    // The earlier own indexer at (`source`, `source_part`) over the rows this
    // source reads.
    let earlier = |source: SublayerIndex, source_part: u64| {
        let same_rows = source_owner(blocks, index, part)
            .zip(source_owner(blocks, source, source_part))
            .is_some_and(|((a, ap, _), (b, bp, _))| a == b && ap == bp);
        (context.shared_sources.is_none() && source < index && same_rows)
            .then(|| attention_at(blocks, source))
            .flatten()
            .and_then(|layer| Some((layer, &layer.history.get(source_part as usize)?.reads)))
            .and_then(|(layer, reads)| match reads {
                HistoryReads::Selected(Selection::Own(owner)) => Some((layer, owner.as_ref())),
                _ => None,
            })
    };
    if let IndexerKeys::Shared { source, part: source_part } = indexer.keys {
        let keys = earlier(source, source_part).filter(|(layer, owner)| {
            matches!(owner.keys, IndexerKeys::Latent { .. })
                && owner.width == indexer.width
                && layer.rotary.of_keys() == attention.rotary.of_keys()
        });
        if keys.is_none() {
            return Err(DefinitionError::new(
                "shared indexer keys name no earlier indexer keeping its own over the same rows",
            ));
        }
    }
    if let Candidates::Restrict { source, part: source_part } = indexer.candidates {
        let pool = earlier(source, source_part).and_then(|(_, owner)| match owner.candidates {
            Candidates::Produce { blocks, block_units } => blocks.checked_mul(block_units),
            _ => None,
        });
        match pool {
            None => {
                return Err(DefinitionError::new(
                    "a restricted indexer names no earlier candidate pool over the same rows",
                ));
            }
            Some(units) if units < indexer.top => {
                return Err(DefinitionError::new(format!(
                    "a candidate pool of {units} units is smaller than its restricted indexer's top {}",
                    indexer.top
                )));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Under hyper-connections a sublayer either mixes and scatters, or is an
/// n-gram input over the raw streams; a single residual uses neither.
fn validate_residual_use(
    sublayer: &Sublayer,
    residual: &ResidualForm,
) -> Result<(), DefinitionError> {
    // A sublayer's site joins its output into the streams: a low-rank site
    // with inject weights, or any Sinkhorn site.
    let mixes = matches!(
        &sublayer.input,
        InputNorm::HyperConnectionMix(HyperConnectionInput {
            site: HyperConnectionWeights::LowRankGated(LowRankGatedSite {
                inject: Some(_),
                ..
            }) | HyperConnectionWeights::Sinkhorn(_),
            ..
        })
    );
    let scatters = matches!(sublayer.output, OutputForm::Scatter);
    let ngram = matches!(sublayer.op, Operator::NGramInput(_));
    let valid = match residual {
        ResidualForm::Single => {
            !ngram && !scatters && !matches!(sublayer.input, InputNorm::HyperConnectionMix(_))
        }
        ResidualForm::HyperConnections(_) if ngram => {
            matches!(sublayer.input, InputNorm::None)
                && matches!(sublayer.output, OutputForm::Residual)
        }
        ResidualForm::HyperConnections(_) => {
            mixes && scatters && !matches!(sublayer.op, Operator::Parallel(_))
        }
    };
    if !valid {
        return Err(DefinitionError::new(format!(
            "{} sublayer disagrees with the residual form",
            sublayer.op.name()
        )));
    }
    Ok(())
}

fn validate_input(input: &InputNorm, context: &Context, role: &str) -> Result<(), DefinitionError> {
    match input {
        InputNorm::Rms(norm) => validate_rms(norm, context.hidden, role),
        InputNorm::RmsUnweighted(norm) => positive_epsilon(norm.epsilon),
        InputNorm::HyperConnectionMix(input) => {
            validate_hyper_connection(&input.site, context, role)?;
            validate_mixed_norm(input.norm.as_ref(), context, role)
        }
        InputNorm::None => Ok(()),
    }
}

/// The low-rank site's stream count and rank.
fn hyper_connection_geometry(context: &Context) -> Option<(u64, u64)> {
    match context.residual {
        ResidualForm::HyperConnections(HyperConnections {
            streams,
            mixing: HyperConnectionMixing::LowRankGated { rank },
        }) => Some((*streams, *rank)),
        _ => None,
    }
}

/// A sublayer site's weights against the decoder's mixing: the variant the
/// mixing names, at its shapes.
fn validate_hyper_connection(
    weights: &HyperConnectionWeights,
    context: &Context,
    role: &str,
) -> Result<(), DefinitionError> {
    let ResidualForm::HyperConnections(form) = context.residual else {
        return Err(DefinitionError::new(format!(
            "{role} mixes streams without hyper-connections"
        )));
    };
    let streams = form.streams;
    let wide = checked_product(&[streams, context.hidden])?;
    match weights {
        HyperConnectionWeights::LowRankGated(site) => {
            validate_low_rank(site, context, role)?;
            // A sublayer's site without inject weights joins nothing, which
            // its residual use refuses (`validate_residual_use`).
            site.inject
                .as_ref()
                .map_or(Ok(()), |inject| expect_shape(inject, &[streams, wide], role))
        }
        HyperConnectionWeights::Sinkhorn(site) => {
            if !matches!(form.mixing, HyperConnectionMixing::Sinkhorn { .. }) {
                return Err(disagreement(role));
            }
            // pre, post and comb.
            let mixes = checked_product(&[streams + 2, streams])?;
            expect_shape(&site.function, &[mixes, wide], role)?;
            expect_shape(&site.base, &[mixes], role)?;
            expect_shape(&site.scale, &[3], role)
        }
    }
}

/// A low-rank site's norm, down and up weights under low-rank mixing.
fn validate_low_rank(
    site: &LowRankGatedSite,
    context: &Context,
    role: &str,
) -> Result<(), DefinitionError> {
    let (streams, rank) = hyper_connection_geometry(context).ok_or_else(|| disagreement(role))?;
    let wide = checked_product(&[streams, context.hidden])?;
    validate_rms(&site.norm, wide, role)?;
    expect_shape(&site.down, &[rank, wide], role)?;
    expect_shape(&site.up, &[wide, rank], role)
}

fn disagreement(role: &str) -> DefinitionError {
    DefinitionError::new(format!(
        "{role}'s site disagrees with the decoder's hyper-connection mixing"
    ))
}

/// The norm of a mixed row, when present, over the hidden width.
fn validate_mixed_norm(
    norm: Option<&RmsNorm>,
    context: &Context,
    role: &str,
) -> Result<(), DefinitionError> {
    norm.map_or(Ok(()), |norm| validate_rms(norm, context.hidden, role))
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
            validate_up(
                &dense.up,
                &[dense.intermediate, hidden],
                "dense feed-forward",
            )?;
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
        Operator::NGramInput(input) => validate_ngram(input, context),
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
                validate_input(&branch.input, context, "branch input norm")?;
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
    match &attention.query {
        QueryProjection::Full(query) => {
            expect_shape(query, &[query_rows, hidden], "attention query")?
        }
        QueryProjection::LowRank { down, norm, up } => {
            let [rank, _] = down.shape[..] else {
                return Err(DefinitionError::new("invalid attention query rank"));
            };
            if rank == 0 {
                return Err(DefinitionError::new("invalid attention query rank"));
            }
            expect_shape(down, &[rank, hidden], "attention query down")?;
            validate_rms(norm, rank, "attention query down norm")?;
            expect_shape(up, &[query_rows, rank], "attention query up")?;
        }
    }
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
    if attention.history.is_empty() {
        return Err(DefinitionError::new("attention reads no history"));
    }
    for (part, source) in attention.history.iter().enumerate() {
        match &source.rows {
            HistoryRows::Own(RowProducer::Projected(rows)) => {
                if part != 0 {
                    return Err(DefinitionError::new(
                        "a projected history source after the first",
                    ));
                }
                validate_projected(rows, attention, hidden)?;
            }
            HistoryRows::Own(RowProducer::Pooled(rows)) => {
                if part == 0 {
                    return Err(DefinitionError::new("a pooled first history source"));
                }
                validate_pooled(rows, width, hidden, "attention pooled rows")?;
            }
            HistoryRows::Shared { .. } => {}
        }
        if let HistoryReads::Selected(Selection::Own(indexer)) = &source.reads {
            validate_indexer(indexer, source, attention, hidden)?;
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
        Rotary::Table {
            pairs,
            layout: _,
            divisors,
            output: _,
        } => {
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
    if let Some(sinks) = &attention.sinks {
        expect_shape(sinks, &[attention.heads], "attention sinks")?;
    }
    match &attention.output {
        OutputProjection::Full(output) => {
            expect_shape(output, &[hidden, head_rows], "attention output")
        }
        OutputProjection::GroupedLowRank { groups, rank, a, b } => {
            let columns = grouped_output_columns(head_rows, *groups)?;
            if *rank == 0 {
                return Err(DefinitionError::new("invalid grouped attention output"));
            }
            let low = checked_product(&[*groups, *rank])?;
            expect_shape(a, &[low, columns], "attention output a")?;
            expect_shape(b, &[hidden, low], "attention output b")
        }
    }
}

fn validate_projected(
    rows: &ProjectedRows,
    attention: &Attention,
    hidden: u64,
) -> Result<(), DefinitionError> {
    let width = attention.width;
    let kv_rows = checked_product(&[attention.kv_heads, width])?;
    expect_shape(&rows.key, &[kv_rows, hidden], "attention key")?;
    match &rows.value {
        ValueSource::Projected(value) => {
            expect_shape(value, &[kv_rows, hidden], "attention value")?
        }
        ValueSource::RawKey => {}
        ValueSource::StoredKey => {
            if !matches!(rows.value_norm, ValueNorm::None) {
                return Err(DefinitionError::new("stored-key values with a value norm"));
            }
        }
    }
    validate_head_norm(&rows.key_norm, width, "attention key norm")?;
    if let ValueNorm::RmsUnweighted(norm) = rows.value_norm {
        positive_epsilon(norm.epsilon)?;
    }
    if rows.domain == (TokenDomain::Window { tokens: 0 }) {
        return Err(DefinitionError::new("invalid attention history domain"));
    }
    // A rounded row is read as both key and value: its value is the key.
    if !matches!(rows.rounding, RowRounding::None)
        && !matches!(rows.value, ValueSource::StoredKey)
    {
        return Err(DefinitionError::new(
            "a rounded history row whose value is not its key",
        ));
    }
    validate_rounding(rows.rounding, width, "attention key rounding")
}

fn validate_pooled(
    rows: &PooledRows,
    width: u64,
    hidden: u64,
    role: &str,
) -> Result<(), DefinitionError> {
    let pooled = match &rows.pool {
        Pool::Identity => rows.rate == 1,
        Pool::Mean => rows.rate >= 2,
        Pool::SoftmaxGated { gate } => {
            expect_shape(gate, &[width, hidden], role)?;
            rows.rate >= 2
        }
    };
    if !pooled {
        return Err(DefinitionError::new(format!("{role}: pool and rate disagree")));
    }
    expect_shape(&rows.projection, &[width, hidden], role)?;
    validate_rms(&rows.norm, width, role)?;
    validate_rounding(rows.rounding, width, role)
}

fn validate_rounding(rounding: RowRounding, width: u64, role: &str) -> Result<(), DefinitionError> {
    match rounding {
        RowRounding::None => Ok(()),
        RowRounding::Block(block) => {
            let floor = block.amax_floor as f32;
            // The element's largest code (`rounding.seismic`).
            let m = match block.element {
                BlockElement::E4M3 => 448.0f32,
                BlockElement::E2M1 => 6.0f32,
            };
            // The floor's scale must be nonzero, or an all-zero run divides
            // 0 by 0. A power-of-two scale is the power of two at or above
            // floor / m, which must be a normal f32: the GPUs' arithmetic
            // flushes a subnormal quotient to zero before its power of two
            // is taken. An E4M3 scale is the
            // nearest E4M3 code of floor / m (ties to even), zero at or below
            // half the least subnormal code 2^-9.
            let scale_vanishes = match block.scale {
                BlockScale::Ue8m0 => floor / m < f32::MIN_POSITIVE,
                BlockScale::E4M3 => floor / m <= 2.0f32.powi(-10),
            };
            if block.block == 0
                || !width.is_multiple_of(block.block)
                || block.amax_floor <= 0.0
                // The kernels take the floor as f32: it must stay a normal one.
                || !floor.is_normal()
                || scale_vanishes
            {
                return Err(DefinitionError::new(format!("{role}: invalid block rounding")));
            }
            Ok(())
        }
    }
}

fn validate_indexer(
    indexer: &Indexer,
    source: &HistorySource,
    attention: &Attention,
    hidden: u64,
) -> Result<(), DefinitionError> {
    let rotary = match &attention.rotary {
        Rotary::None => 0,
        Rotary::Interleaved { width, .. } => *width,
        Rotary::Table { pairs, .. } => 2 * pairs.len() as u64,
    };
    if [indexer.top, indexer.heads, indexer.width].contains(&0)
        || rotary > indexer.width
        || !indexer.scale.is_finite()
        || indexer.scale <= 0.0
    {
        return Err(DefinitionError::new("invalid selected history geometry"));
    }
    let input = match (&indexer.query.input, &attention.query) {
        (IndexerInput::Hidden, _) => hidden,
        (IndexerInput::QueryRank, QueryProjection::LowRank { down, .. }) => down.shape[0],
        (IndexerInput::QueryRank, QueryProjection::Full(_)) => {
            return Err(DefinitionError::new(
                "indexer reading the query rank of a full-rank query",
            ));
        }
    };
    expect_shape(
        &indexer.query.projection,
        &[checked_product(&[indexer.heads, indexer.width])?, input],
        "indexer query",
    )?;
    validate_head_norm(&indexer.query.norm, indexer.width, "indexer query norm")?;
    validate_rounding(indexer.query.rounding, indexer.width, "indexer query rounding")?;
    if let HeadWeights::Projected(weight) = &indexer.head_weights {
        expect_shape(weight, &[indexer.heads, hidden], "indexer head weights")?;
    }
    // A pool is produced over a pooled source's own rows by the indexer that
    // keeps their keys; its restrictions are checked against it
    // (`validate_shared_sources`).
    if let Candidates::Produce { blocks, block_units } = indexer.candidates {
        let owns_keys = matches!(indexer.keys, IndexerKeys::Latent { .. });
        let pooled = matches!(source.rows, HistoryRows::Own(RowProducer::Pooled(_)));
        if blocks == 0 || block_units == 0 || !owns_keys || !pooled {
            return Err(DefinitionError::new(
                "a candidate pool is produced over its own pooled rows and keys, in blocks of units",
            ));
        }
    }
    match (&indexer.keys, &source.rows) {
        // Pooled keys over a token source: each covers its group of tokens.
        (
            IndexerKeys::Pooled(keys),
            HistoryRows::Own(RowProducer::Projected(ProjectedRows {
                domain: TokenDomain::Token,
                ..
            })),
        ) => validate_pooled(keys, indexer.width, hidden, "indexer keys"),
        // One key per row of the layer's own pooled source.
        (
            IndexerKeys::Latent {
                projection,
                norm,
                rounding,
            },
            HistoryRows::Own(RowProducer::Pooled(_)),
        ) => {
            expect_shape(projection, &[indexer.width, attention.width], "indexer keys")?;
            validate_rms(norm, indexer.width, "indexer key norm")?;
            validate_rounding(*rounding, indexer.width, "indexer key rounding")
        }
        // Another layer's keys over the rows it owns and this layer shares
        // (`validate_shared_sources`).
        (IndexerKeys::Shared { .. }, HistoryRows::Shared { .. }) => Ok(()),
        _ => Err(DefinitionError::new(
            "indexer keys disagree with the rows they select",
        )),
    }
}

fn validate_ngram(input: &NGramInput, context: &Context) -> Result<(), DefinitionError> {
    let ResidualForm::HyperConnections(HyperConnections { streams, .. }) = context.residual else {
        return Err(DefinitionError::new("n-gram input without hyper-connections"));
    };
    let hash = &input.hash;
    let heads = hash.heads()?;
    let rows = input.table.shape.first().copied().unwrap_or(0);
    let heads_in_table = hash.sizes.iter().zip(&hash.offsets).all(|(size, offset)| {
        *size > 0 && offset.checked_add(*size).is_some_and(|end| end <= rows)
    });
    let mapped = hash
        .token_map
        .as_ref()
        .is_none_or(|map| map.len() as u64 == context.vocabulary);
    let hashed = hash.hashed_vocabulary(context.vocabulary);
    // Every product `id · multiplier` of a hashed id is exact in u64.
    let exact = hash
        .multipliers
        .iter()
        .all(|multiplier| *multiplier > 0 && multiplier.checked_mul(hashed).is_some());
    let mixing = input
        .mixing
        .as_ref()
        .is_none_or(|mixing| mixing.kernel >= 2 && mixing.dilation > 0);
    if heads == 0
        || input.width == 0
        || hash.multipliers.len() as u64 != hash.context + 1
        || hash.sizes.len() as u64 != heads
        || hash.offsets.len() as u64 != heads
        || !heads_in_table
        || !mapped
        || !exact
        || hash.fill >= hashed
        || hash.boundary.is_some_and(|boundary| boundary >= context.vocabulary)
        || !mixing
    {
        return Err(DefinitionError::new("invalid n-gram input geometry"));
    }
    positive_epsilon(input.epsilon)?;
    let embedding = checked_product(&[heads, input.width])?;
    let wide = checked_product(&[*streams, context.hidden])?;
    expect_shape(&input.table, &[rows, input.width], "n-gram table")?;
    expect_shape(&input.key, &[wide, embedding], "n-gram key")?;
    expect_shape(&input.value, &[context.hidden, embedding], "n-gram value")?;
    match &input.gate {
        NGramGate::Norms { key, query } => {
            expect_shape(key, &[wide], "n-gram key norm")?;
            expect_shape(query, &[wide], "n-gram query norm")?;
        }
        NGramGate::Product(weight) => expect_shape(weight, &[wide], "n-gram gate")?,
    }
    match &input.mixing {
        None => Ok(()),
        Some(mixing) => {
            expect_shape(&mixing.norm, &[wide], "n-gram convolution norm")?;
            expect_shape(&mixing.taps, &[wide, mixing.kernel], "n-gram convolution")
        }
    }
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
    expect_shape(
        &delta.alpha,
        &[delta.value_heads, hidden],
        "gated delta alpha",
    )?;
    expect_shape(
        &delta.beta,
        &[delta.value_heads, hidden],
        "gated delta beta",
    )?;
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
    if let Some(bias) = &router.media_bias {
        expect_shape(bias, &[routed.experts], "media-row expert selection bias")?;
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
        validate_up(&shared.up, &[shared.intermediate, hidden], "shared expert")?;
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
    context: &Context,
    role: &str,
) -> Result<(), DefinitionError> {
    let hyper_connections = matches!(context.residual, ResidualForm::HyperConnections(_));
    match norm {
        ExitNorm::Rms(norm) if !hyper_connections => validate_rms(norm, context.hidden, role),
        ExitNorm::Layer(norm) if !hyper_connections => {
            expect_shape(&norm.weight, &[context.hidden], role)?;
            positive_epsilon(norm.epsilon)
        }
        ExitNorm::HyperConnectionHead(exit) if hyper_connections => {
            match &exit.site {
                ExitSite::Own(site) => {
                    validate_low_rank(site, context, role)?;
                    if site.inject.is_some() {
                        return Err(DefinitionError::new(format!(
                            "{role} injects into no stream"
                        )));
                    }
                }
                ExitSite::PreviousSite => {
                    if !matches!(
                        context.residual,
                        ResidualForm::HyperConnections(HyperConnections {
                            mixing: HyperConnectionMixing::Sinkhorn { .. },
                            ..
                        })
                    ) {
                        return Err(DefinitionError::new(format!(
                            "{role} reads the previous site's coefficients without Sinkhorn mixing"
                        )));
                    }
                }
            }
            validate_mixed_norm(exit.norm.as_ref(), context, role)
        }
        _ => Err(DefinitionError::new(format!(
            "{role} disagrees with the residual form"
        ))),
    }
}

fn positive_epsilon(epsilon: f64) -> Result<(), DefinitionError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(DefinitionError::new("invalid normalization epsilon"));
    }
    Ok(())
}
