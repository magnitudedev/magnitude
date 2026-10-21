//! The sublayer operators this executor implements (model-family plan §3.2).
//!
//! A model definition describes any decoder the family contract can express.
//! This module is the one place that decides which of those forms the
//! executor runs, and rejects every other form with a typed plan error, so
//! assessment reports it `Incompatible` rather than failing on a device.

pub(crate) mod attention;
pub(crate) mod block;
pub(crate) mod dense_ffn;
pub(crate) mod draft;
pub(crate) mod gated_delta;
pub(crate) mod ngram;
pub(crate) mod output;
pub(crate) mod parallel;
pub(crate) mod per_layer;
pub(crate) mod routed;
pub(crate) mod short_conv;
pub(crate) mod state_space;
pub(crate) mod streams;
pub(crate) mod vision;

use crate::SublayerTail;
use crate::error::PlanError;
use magnitude_family_contracts::{
    Attention, Block, BranchOutput, Decoder, DenseFfn, ExitNorm, GatedDelta, Head, HeadBlock,
    HeadChain, HeadComponent, HeadHistory, HeadInput, HeadReadout, HyperConnectionWeights,
    InputNorm, KeyValue, ModelDefinition, Operator, OutputForm, ProposalReadoutPolicy,
    ResidualForm, RmsNorm, RoutedFfn, RouterInput, Sublayer, SublayerIndex, VocabularyProjection,
    WeightDescriptor, WeightKind, WeightRole, WeightScope,
};
use magnitude_family_contracts::{HistoryReads, ShortConv, StateSpace};
use seismic::Element;

/// How a sublayer's output joins the residual (`scatter`): into `streams`
/// hyper-connection streams by a mix site's inject values (`[M, streams]`),
/// or into one stream (`[M, 0]`, no inject).
pub(crate) struct Join {
    /// The mix site's inject values; none for one stream.
    pub values: Option<seismic::WorkflowTensor>,
    pub streams: u64,
    pub injected: bool,
    /// The class dimension the joined rows follow in their graph, if any.
    pub class: Option<&'static str>,
}

impl Join {
    /// The join of a single residual stream whose rows follow the graph's
    /// class dimension `class` (none: fixed rows).
    pub(crate) fn single(class: Option<&'static str>) -> Self {
        Self {
            values: None,
            streams: 1,
            injected: false,
            class,
        }
    }

    /// The inject port an output entry over `rows` rows binds: the mix
    /// site's values, or for one stream an empty `[rows, 0]` constant, made
    /// only by an entry that binds it (a row-class constant,
    /// `block::class_constants_of`).
    pub(crate) fn inject<G: crate::programs::graph::draft::GraphDraft>(
        &self,
        graph: &mut G,
        rows: u64,
        constants: &mut Vec<crate::programs::native_constants::GraphConstant>,
    ) -> Result<seismic::WorkflowTensor, crate::programs::graph::GraphError> {
        match &self.values {
            Some(values) => Ok(values.clone()),
            None => {
                crate::programs::native_constants::GraphConstant::absent_inject(
                    graph, rows, self.class, constants,
                )
            }
        }
    }

    /// The output entries' join dimensions `SN` and `IN`.
    pub(crate) fn dimensions(&self) -> [(&'static str, u64); 2] {
        [
            ("SN", self.streams),
            ("IN", if self.injected { self.streams } else { 0 }),
        ]
    }
}

/// A sublayer's input as its consumer entry reads it: `rows` formed per
/// `form` (`form_input`), the residual under the sublayer's input norm or a
/// mix site's partials.
#[derive(Clone, Copy)]
pub(crate) struct SublayerInput<'t> {
    pub rows: &'t seismic::WorkflowTensor,
    pub form: InputForm,
}

impl<'t> SublayerInput<'t> {
    /// The single residual stream `rows` under the sublayer's input norm.
    pub(crate) fn single(rows: &'t seismic::WorkflowTensor) -> Self {
        Self {
            rows,
            form: InputForm::SINGLE,
        }
    }
}

/// How a sublayer's consumer entry forms its input row (`form_input`):
/// the sum of `parts` partial rows, normalized by the sublayer's input norm
/// when `normalized`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InputForm {
    pub parts: u64,
    pub normalized: bool,
}

impl InputForm {
    /// One residual row under the sublayer's input norm.
    pub(crate) const SINGLE: Self = Self {
        parts: 1,
        normalized: true,
    };

    /// The decoder readout's input form: the exit site's partial rows, one
    /// per hyper-connection stream, unnormalized; or the final row under the
    /// exit norm.
    pub(crate) fn exit(decoder: &Decoder) -> Self {
        match (&decoder.exit.norm, &decoder.residual) {
            (ExitNorm::HyperConnectionHead(_), ResidualForm::HyperConnections(form)) => Self {
                parts: form.streams,
                normalized: false,
            },
            _ => Self::SINGLE,
        }
    }

    /// The consumer entries' input dimensions `PT` and `NM` over rows of
    /// `width`.
    pub(crate) fn dimensions(&self, width: u64) -> [(&'static str, u64); 2] {
        [("PT", self.parts), ("NM", self.norm_extent(width))]
    }

    /// The extent of the input norm over rows of `width`: 0 when the form
    /// normalizes nothing.
    pub(crate) fn norm_extent(&self, width: u64) -> u64 {
        if self.normalized { width } else { 0 }
    }
}

/// An operator module's sink for the weights it binds, by semantic kind.
pub(super) type WeightPush<'p, 'a> = dyn FnMut(WeightKind, &'a WeightDescriptor) + 'p;

/// The tail of a sublayer with `output`; `lookup` resolves the planned
/// element of a role in the sublayer's scope.
pub(crate) fn tail(
    output: &OutputForm,
    lookup: &impl Fn(WeightKind) -> Result<Element, PlanError>,
) -> Result<SublayerTail, PlanError> {
    match output {
        // A scatter is the residual output joining the streams (`Join`).
        OutputForm::Residual | OutputForm::Scatter => Ok(SublayerTail::Residual),
        OutputForm::PostNorm(_) => Ok(SublayerTail::PostNorm {
            norm: lookup(WeightKind::PostNorm)?,
            scaled: false,
        }),
        OutputForm::ScaledPostNorm { .. } => Ok(SublayerTail::PostNorm {
            norm: lookup(WeightKind::PostNorm)?,
            scaled: true,
        }),
    }
}

/// The weights of the decoder's exit norm (`WeightScope::Target`): its norm
/// weight, or a hyper-connection exit site's weights in the order its mix
/// consumes them.
pub(crate) fn exit_weights(decoder: &Decoder) -> Vec<(WeightRole, &WeightDescriptor)> {
    match &decoder.exit.norm {
        ExitNorm::HyperConnectionHead(site) => {
            let mut weights = Vec::new();
            hyper_connection_weights(site, WeightScope::Target, &mut weights);
            weights
        }
        norm => vec![(
            role(WeightScope::Target, WeightKind::OutputNorm),
            norm.weight(),
        )],
    }
}

/// Every weight of the decoder's sublayers with its semantic role, in model
/// order. Entry and exit weights are the caller's (`WeightScope::Target`).
pub(crate) fn decoder_weights(decoder: &Decoder) -> Vec<(WeightRole, &WeightDescriptor)> {
    let mut weights = Vec::new();
    for (index, sublayer) in decoder.sublayers() {
        sublayer_weights(
            sublayer,
            WeightScope::TargetSublayer(index),
            &mut weights,
            |branch| WeightScope::TargetBranch {
                sublayer: index,
                branch,
            },
        );
    }
    weights
}

/// The row rules of the decoder's n-gram inputs, by their blocks.
pub(crate) fn row_rules(decoder: &Decoder) -> Vec<(usize, &magnitude_family_contracts::NGramHash)> {
    decoder
        .sublayers()
        .filter_map(|(index, sublayer)| match &sublayer.op {
            Operator::NGramInput(input) => Some((index.block as usize, &input.hash)),
            _ => None,
        })
        .collect()
}

/// The preceding tokens the decoder's row rules read: 0 without one.
pub(crate) fn row_rule_context(decoder: &Decoder) -> usize {
    row_rules(decoder)
        .into_iter()
        .map(|(_, hash)| hash.context as usize)
        .max()
        .unwrap_or(0)
}

/// Every weight of a draft head, in model order.
pub(crate) fn head_weights(head: &Head) -> Result<Vec<(WeightRole, &WeightDescriptor)>, PlanError> {
    let mut weights = Vec::new();
    // `WeightScope::HeadSublayer` counts the decoder blocks of every head
    // block in order.
    let mut decoder_block = 0u32;
    for (block, head_block) in head.blocks.iter().enumerate() {
        let block = u32::try_from(block).map_err(|_| PlanError::Arithmetic("head block index"))?;
        let scope = WeightScope::HeadBlock(block);
        let role = |kind| WeightRole { scope, kind };
        match &head_block.input {
            HeadInput::Combine {
                embedding_norm,
                hidden_norm,
                combine,
            } => {
                weights.push((role(WeightKind::HeadEmbeddingNorm), &embedding_norm.weight));
                weights.push((role(WeightKind::HeadHiddenNorm), &hidden_norm.weight));
                weights.push((role(WeightKind::HeadCombine), combine));
            }
            HeadInput::Projected { pre, .. } => {
                weights.push((role(WeightKind::HeadInputProjection), pre));
            }
        }
        for value in &head_block.blocks {
            for (sublayer, value) in value.sublayers.iter().enumerate() {
                let index = SublayerIndex {
                    block: decoder_block,
                    sublayer: u32::try_from(sublayer)
                        .map_err(|_| PlanError::Arithmetic("head sublayer index"))?,
                };
                sublayer_weights(
                    value,
                    WeightScope::HeadSublayer(index),
                    &mut weights,
                    |branch| WeightScope::HeadBranch {
                        sublayer: index,
                        branch,
                    },
                );
            }
            decoder_block = decoder_block
                .checked_add(1)
                .ok_or(PlanError::Arithmetic("head block index"))?;
        }
        match &head_block.output_norm {
            ExitNorm::HyperConnectionHead(site) => hyper_connection_weights(site, scope, &mut weights),
            norm => weights.push((role(WeightKind::OutputNorm), norm.weight())),
        }
        if let Some(table) = head_block.readout.own_table() {
            weights.push((role(WeightKind::Output), table));
        }
        if let ProposalReadoutPolicy::Clustered { centroids, .. } = &head_block.readout.policy {
            weights.push((role(WeightKind::OutputCentroids), centroids));
        }
        if let HeadChain::Projected { post } = &head_block.chain {
            weights.push((role(WeightKind::HeadChainProjection), post));
        }
    }
    Ok(weights)
}

/// A head block over its own causal history: combine the successor embedding
/// with conditioning, execute one decoder block, and share the target readout.
/// Conditioning is the normalized output or the raw residual streams.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CausalHeadBlock<'a> {
    pub embedding_norm: &'a RmsNorm,
    pub hidden_norm: &'a RmsNorm,
    pub block: &'a Block,
    pub residual: &'a ResidualForm,
    pub chain: &'a HeadChain,
}

impl<'a> CausalHeadBlock<'a> {
    pub fn of(head: &Head, block: &'a HeadBlock) -> Result<Self, PlanError> {
        match (
            head.history,
            &block.input,
            block.blocks.as_slice(),
            &block.readout,
            &block.chain,
        ) {
            (
                HeadHistory::Own,
                HeadInput::Combine {
                    embedding_norm,
                    hidden_norm,
                    ..
                },
                [decoder_block],
                HeadReadout {
                    projection: VocabularyProjection::Target,
                    policy: ProposalReadoutPolicy::Full,
                },
                HeadChain::Output | HeadChain::Residual,
            ) => Ok(Self {
                embedding_norm,
                hidden_norm,
                block: decoder_block,
                residual: &block.residual,
                chain: &block.chain,
            }),
            _ => Err(PlanError::Unsupported("draft head form")),
        }
    }
}

/// The draft head forms this executor runs.
#[derive(Clone, Copy, Debug)]
pub(crate) enum HeadForm<'a> {
    /// A head over its own history: every head block
    /// is an [`CausalHeadBlock`].
    Causal,
    /// A head of its own width reading the target's history.
    Shared(SharedHead<'a>),
}

/// A draft head that reads target history (`HeadHistory::Target`, stored in
/// the draft component): one head block whose input is the target's scaled
/// embedding beside a hidden row projected into the head's width, whose
/// decoder blocks attend only the target's accepted history, which reads out
/// through its own full-vocabulary table without a soft cap and chains
/// through a projection back to the target's width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedHead<'a> {
    pub width: u64,
    pub blocks: &'a [Block],
    pub output_norm: &'a RmsNorm,
    pub readout: &'a HeadReadout,
}

impl SharedHead<'_> {
    /// The target sublayer whose history the head block `index` reads.
    pub fn source(&self, index: usize) -> Result<SublayerIndex, PlanError> {
        let block = self
            .blocks
            .get(index)
            .ok_or(PlanError::Topology("draft head block index"))?;
        match &paired_block(block)?.mixer {
            Mixer::Attention(attention) => match &attention.key_value {
                KeyValue::Shared { source } => Ok(*source),
                KeyValue::Owned { .. } => Err(PlanError::Topology(
                    "a head reading target history owns none",
                )),
            },
            _ => Err(PlanError::Unsupported("draft head mixer")),
        }
    }
}

/// The form of `head` among those this executor runs.
pub(crate) fn head_form<'a>(head: &'a Head, decoder: &Decoder) -> Result<HeadForm<'a>, PlanError> {
    match head.history {
        HeadHistory::Own => {
            for block in &head.blocks {
                CausalHeadBlock::of(head, block)?;
            }
            Ok(HeadForm::Causal)
        }
        HeadHistory::Target => {
            let [block] = head.blocks.as_slice() else {
                return Err(PlanError::Unsupported(
                    "chained draft heads reading target history",
                ));
            };
            match (
                head.component,
                &block.input,
                &block.readout,
                &block.chain,
                &block.output_norm,
            ) {
                (
                    HeadComponent::Draft,
                    HeadInput::Projected {
                        embedding_scale, ..
                    },
                    HeadReadout {
                        projection: VocabularyProjection::Own { .. },
                        policy: _,
                    },
                    HeadChain::Projected { .. },
                    ExitNorm::Rms(output_norm),
                ) => {
                    // `embedding_rows` applies the target's entry transform.
                    if *embedding_scale != decoder.entry.scale {
                        return Err(PlanError::Unsupported(
                            "a draft head scaling the target's embedding differently",
                        ));
                    }
                    Ok(HeadForm::Shared(SharedHead {
                        width: block.width(decoder),
                        blocks: &block.blocks,
                        output_norm,
                        readout: &block.readout,
                    }))
                }
                _ => Err(PlanError::Unsupported("draft head form")),
            }
        }
    }
}

fn sublayer_weights<'a>(
    sublayer: &'a Sublayer,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, &'a WeightDescriptor)>,
    branch_scope: impl Fn(u32) -> WeightScope,
) {
    input_weights(&sublayer.input, scope, weights);
    if let Operator::Parallel(branches) = &sublayer.op {
        for (branch, value) in branches.iter().enumerate() {
            let scope = branch_scope(branch as u32);
            input_weights(&value.input, scope, weights);
            operator_weights(&value.op, scope, weights);
            if let BranchOutput::Norm(norm) = &value.output {
                weights.push((role(scope, WeightKind::PostNorm), &norm.weight));
            }
        }
    } else {
        operator_weights(&sublayer.op, scope, weights);
    }
    match &sublayer.output {
        OutputForm::Residual | OutputForm::Scatter => {}
        OutputForm::PostNorm(norm) => {
            weights.push((role(scope, WeightKind::PostNorm), &norm.weight))
        }
        OutputForm::ScaledPostNorm { norm, layer_scale } => {
            weights.push((role(scope, WeightKind::PostNorm), &norm.weight));
            weights.push((role(scope, WeightKind::LayerScale), layer_scale));
        }
    }
}

fn role(scope: WeightScope, kind: WeightKind) -> WeightRole {
    WeightRole { scope, kind }
}

fn input_weights<'a>(
    input: &'a InputNorm,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, &'a WeightDescriptor)>,
) {
    match input {
        InputNorm::Rms(norm) => weights.push((role(scope, WeightKind::InputNorm), &norm.weight)),
        InputNorm::HyperConnectionMix(site) => hyper_connection_weights(site, scope, weights),
        InputNorm::RmsUnweighted(_) | InputNorm::None => {}
    }
}

/// A hyper-connection site's weights in the order its mix consumes them.
pub(crate) fn hyper_connection_weights<'a>(
    site: &'a HyperConnectionWeights,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, &'a WeightDescriptor)>,
) {
    weights.push((
        role(scope, WeightKind::HyperConnectionNorm),
        &site.norm.weight,
    ));
    weights.push((role(scope, WeightKind::HyperConnectionDown), &site.down));
    weights.push((role(scope, WeightKind::HyperConnectionUp), &site.up));
    if let Some(inject) = &site.inject {
        weights.push((role(scope, WeightKind::HyperConnectionInject), inject));
    }
}

/// The weights an operator binds, by semantic kind. The order is the order
/// the operator's kernels consume them.
fn operator_weights<'a>(
    op: &'a Operator,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, &'a WeightDescriptor)>,
) {
    let mut push = |kind, weight| weights.push((role(scope, kind), weight));
    match op {
        Operator::Attention(operator) => attention::weights(operator, &mut push),
        Operator::LatentAttention(_) => {}
        Operator::GatedDelta(operator) => gated_delta::weights(operator, &mut push),
        Operator::ShortConv(operator) => short_conv::weights(operator, &mut push),
        Operator::StateSpace(operator) => state_space::weights(operator, &mut push),
        Operator::DenseFfn(operator) => dense_ffn::weights(operator, &mut push),
        Operator::RoutedFfn(operator) => routed::weights(operator, &mut push),
        Operator::PerLayerInput(operator) => per_layer::weights(operator, &mut push),
        // The table is a gathered table (`gathered_table`).
        Operator::NGramInput(operator) => {
            push(WeightKind::NGramTable, &operator.table);
            push(WeightKind::NGramKey, &operator.key);
            push(WeightKind::NGramValue, &operator.value);
            push(WeightKind::NGramKeyNorm, &operator.key_norm.weight);
            push(WeightKind::NGramQueryNorm, &operator.query_norm.weight);
            push(
                WeightKind::NGramConvolutionNorm,
                &operator.convolution_norm.weight,
            );
            push(WeightKind::NGramConvolution, &operator.convolution);
        }
        Operator::Parallel(_) => {}
    }
}

/// Roles consumed by fixed-f32 kernel arguments. The resident representation
/// is part of the semantic kernel ABI, not a blanket model-wide preference:
/// every other weight follows the component activation dtype (or its
/// admitted packed format).
pub(crate) fn is_fixed_dense_role(kind: WeightKind) -> bool {
    matches!(
        kind,
        WeightKind::QueryNorm
            | WeightKind::KeyNorm
            | WeightKind::IndexerQueryNorm
            | WeightKind::IndexerKeyNorm
            | WeightKind::RecurrentConvolution
            | WeightKind::RecurrentConvolutionBias
            | WeightKind::NGramConvolution
            | WeightKind::RecurrentDecay
            | WeightKind::RecurrentTimeBias
            | WeightKind::StateSpaceSkip
            | WeightKind::StateSpaceNorm
            | WeightKind::RouterSelectionBias
            | WeightKind::ExpertScale
            | WeightKind::SharedRouter
            | WeightKind::ConvolutionBase
            | WeightKind::AttentionSinks
    ) || matches!(kind, WeightKind::Corrector(weight) if draft::corrector_reads_f32(weight))
}

/// The dense element a role becomes resident in.
pub(crate) fn resident_dtype(kind: WeightKind, activation: seismic::DType) -> seismic::DType {
    if is_fixed_dense_role(kind) {
        seismic::DType::F32
    } else {
        activation
    }
}

/// A mixer sublayer operator.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Mixer<'a> {
    Attention(&'a Attention),
    GatedDelta(&'a GatedDelta),
    StateSpace(&'a StateSpace),
    ShortConv(&'a ShortConv),
}

/// A feed-forward sublayer operator.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FeedForward<'a> {
    Dense(&'a DenseFfn),
    /// Qwen's fused routed form (`routed::is_fused`).
    Routed(&'a RoutedFfn),
    /// Every other admitted routed form (`routed`).
    GeneralRouted(&'a RoutedFfn),
    /// A dense branch beside a general routed branch (Gemma 4 26B).
    Parallel(parallel::DenseBesideRouted<'a>),
}

/// A block of one pre-normalized mixer sublayer, followed by one
/// pre-normalized feed-forward sublayer unless the block is a lone mixer
/// (Nemotron-H), each adding its output to the residual, directly or
/// normalized first (`OutputForm::PostNorm`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairedBlock<'a> {
    pub mixer_input: SublayerSource<'a>,
    pub mixer: Mixer<'a>,
    pub mixer_output: &'a OutputForm,
    pub feed_forward: Option<FeedForwardSublayer<'a>>,
    /// The per-layer input sublayer after the feed-forward (Gemma E2B/E4B):
    /// unnormalized input, a post-norm tail.
    pub per_layer: Option<per_layer::PerLayerSublayer<'a>>,
    /// The n-gram input after the feed-forward, at the head of the next
    /// block's streams (qwen4exp).
    pub ngram: Option<ngram::NGramSublayer<'a>>,
}

/// What a sublayer reads: the residual under its RMS input norm, or a
/// hyper-connection mix site over the residual streams.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SublayerSource<'a> {
    Norm(&'a RmsNorm),
    Mix(&'a HyperConnectionWeights),
}

impl SublayerSource<'_> {
    /// The epsilon of the source's norm.
    pub fn epsilon(&self) -> f64 {
        match self {
            Self::Norm(norm) => norm.epsilon,
            Self::Mix(mix) => mix.norm.epsilon,
        }
    }

    /// The planned element of a role of the sublayer, from `lookup`: a mix
    /// site's sublayer has no input norm, and its consumer binds an empty
    /// F32 norm port.
    pub fn lookup(
        &self,
        kind: WeightKind,
        lookup: &impl Fn(WeightKind) -> Result<Element, PlanError>,
    ) -> Result<Element, PlanError> {
        match (self, kind) {
            (Self::Mix(_), WeightKind::InputNorm) => Ok(Element::f32()),
            _ => lookup(kind),
        }
    }

    /// How the sublayer's consumer entry forms its input over `streams`
    /// residual streams: one normalized row, or the mix site's partials.
    pub fn form(&self, streams: u64) -> InputForm {
        match self {
            Self::Norm(_) => InputForm::SINGLE,
            Self::Mix(_) => InputForm {
                parts: streams,
                normalized: false,
            },
        }
    }
}

/// The feed-forward sublayer of a [`PairedBlock`]. Its source is absent
/// only for parallel branches, which normalize their own inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FeedForwardSublayer<'a> {
    pub input: Option<SublayerSource<'a>>,
    pub op: FeedForward<'a>,
    pub output: &'a OutputForm,
}

impl FeedForwardSublayer<'_> {
    /// The epsilons of every norm the sublayer's kernels share one epsilon
    /// with: its source's, or its branches' input and output norms.
    pub fn epsilons(&self) -> Vec<f64> {
        match self.op {
            FeedForward::Parallel(branches) => {
                branches.norms().iter().map(|norm| norm.epsilon).collect()
            }
            _ => self.input.iter().map(|input| input.epsilon()).collect(),
        }
    }
}

/// Which state a mixer keeps, which decides the launch class axis of its
/// block's graphs: history segments for attention, request slots for a
/// recurrent bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MixerKind {
    Attention,
    Recurrent,
}

impl Mixer<'_> {
    pub fn kind(&self) -> MixerKind {
        match self {
            Self::Attention(_) => MixerKind::Attention,
            Self::GatedDelta(_) | Self::StateSpace(_) | Self::ShortConv(_) => MixerKind::Recurrent,
        }
    }

    /// The components the mixer keeps in the state store's recurrent bank,
    /// consecutive in model order.
    pub fn bank_components(&self) -> usize {
        match self {
            Self::Attention(_) => 0,
            // Window, delta or state, tape.
            Self::GatedDelta(_) | Self::StateSpace(_) => 3,
            // The window alone: its rows are the tape.
            Self::ShortConv(_) => 1,
        }
    }
}

/// The index of block `index`'s first recurrent bank component: the
/// components of every earlier block come first.
pub(crate) fn bank_component_index(blocks: &[Block], index: usize) -> Result<usize, PlanError> {
    blocks[..index].iter().try_fold(0, |components, block| {
        Ok(components + paired_block(block)?.bank_components())
    })
}

impl PairedBlock<'_> {
    /// The components the block keeps in the state store's recurrent bank,
    /// consecutive in model order: its mixer's, then its n-gram input's.
    pub fn bank_components(&self) -> usize {
        self.mixer.bank_components() + self.ngram.map_or(0, |ngram| ngram.bank_components())
    }

    /// The one normalization epsilon the block's kernels take.
    pub fn epsilon(&self) -> f64 {
        self.mixer_input.epsilon()
    }

    /// Every numerical parameter of the block apart from its weights. Blocks
    /// with equal keys and equal resident weight shapes seal equal graphs.
    pub fn shape_key(&self) -> Result<String, PlanError> {
        let mixer = match self.mixer {
            Mixer::Attention(operator) => attention::shape_key(operator, self.epsilon())?,
            Mixer::GatedDelta(operator) => gated_delta::shape_key(operator),
            Mixer::StateSpace(operator) => state_space::shape_key(operator),
            Mixer::ShortConv(operator) => short_conv::shape_key(operator),
        };
        let feed_forward = self.feed_forward.map(|feed_forward| {
            let op = match feed_forward.op {
                FeedForward::Dense(operator) => dense_ffn::shape_key(operator),
                FeedForward::Routed(operator) => routed::fused_shape_key(operator),
                FeedForward::GeneralRouted(routed) => format!(
                    "general routed {:?}",
                    routed::GeneralRoutedShape::of(0, routed)
                ),
                FeedForward::Parallel(branches) => parallel::shape_key(branches),
            };
            format!("{op} {:?}", post_norm_epsilon(feed_forward.output))
        });
        let per_layer = self.per_layer.map(per_layer::shape_key);
        let ngram = self.ngram.map(ngram::shape_key);
        Ok(format!(
            "{mixer} {:?}; {feed_forward:?}; {per_layer:?}; {ngram:?}; {}",
            post_norm_epsilon(self.mixer_output),
            self.epsilon()
        ))
    }
}

/// The epsilon of a sublayer's post-norm, when its output is normalized
/// before it joins the residual.
pub(crate) fn post_norm_epsilon(output: &OutputForm) -> Option<f64> {
    match output {
        OutputForm::PostNorm(norm) | OutputForm::ScaledPostNorm { norm, .. } => Some(norm.epsilon),
        OutputForm::Residual | OutputForm::Scatter => None,
    }
}

/// Admit a definition: every form it uses must be one this executor runs.
/// Its draft head counts only when `head` selects it: an unselected head's
/// form does not bear on the model.
pub(crate) fn admit(definition: &ModelDefinition, head: bool) -> Result<(), PlanError> {
    if let Some(form) = definition.deferred_forms().first() {
        return Err(PlanError::Deferred(*form));
    }
    let decoder = &definition.decoder;
    // Residual streams mix only low rank and gated (`streams`).
    streams::StreamsGeometry::of(decoder)?;
    // `embedding_rows` scales (`embedding_transform`) and normalizes text
    // rows. Media rows enter as their projector leaves them: a family whose
    // text rows are normalized at entry normalizes its media rows in its
    // projector (Muse's `perception_emb_norm`).
    attention::admit_media(decoder, definition.vision.is_some())?;
    // The final norm, or the exit site's (the contract pairs each with its
    // residual form).
    let exit = match &decoder.exit.norm {
        ExitNorm::Rms(norm) => norm,
        ExitNorm::HyperConnectionHead(site) => &site.norm,
        _ => return Err(PlanError::Unsupported("readout normalization")),
    };
    for block in &decoder.blocks {
        let paired = paired_block(block)?;
        if paired.feed_forward.is_some_and(|feed_forward| {
            feed_forward
                .epsilons()
                .iter()
                .any(|epsilon| *epsilon != paired.epsilon())
        }) || exit.epsilon != paired.epsilon()
        {
            return Err(PlanError::Unsupported(
                "normalization epsilons differing within a decoder",
            ));
        }
    }
    if let Some(head) = definition.head.as_ref().filter(|_| head) {
        let form = head_form(head, decoder)?;
        if let HeadForm::Shared(shared) = form {
            admit_shared_head(&shared)?;
            // Its kernels are prepared with the decoder's one epsilon.
            if shared.output_norm.epsilon != exit.epsilon {
                return Err(PlanError::Unsupported(
                    "a draft head normalizing with another epsilon than its target",
                ));
            }
        }
        for head_block in head
            .blocks
            .iter()
            .filter(|_| matches!(form, HeadForm::Causal))
        {
            let block = CausalHeadBlock::of(head, head_block)?;
            let paired = paired_block(block.block)?;
            if !matches!(paired.mixer, Mixer::Attention(_)) {
                return Err(PlanError::Unsupported("draft head mixer"));
            }
            let Some(feed_forward) = paired.feed_forward else {
                return Err(PlanError::Unsupported(
                    "draft head block without feed-forward",
                ));
            };
            if matches!(
                feed_forward.op,
                FeedForward::GeneralRouted(_) | FeedForward::Parallel(_)
            ) {
                return Err(PlanError::Unsupported(
                    "draft head routed feed-forward form",
                ));
            }
            let expected = match block.residual {
                ResidualForm::Single => OutputForm::Residual,
                ResidualForm::HyperConnections(_) => OutputForm::Scatter,
            };
            if [paired.mixer_output, feed_forward.output]
                .iter()
                .any(|output| **output != expected)
            {
                return Err(PlanError::Unsupported("draft head sublayer output form"));
            }
            let output_epsilon = match &head_block.output_norm {
                ExitNorm::Rms(output) => output.epsilon,
                ExitNorm::HyperConnectionHead(site) => site.norm.epsilon,
                _ => return Err(PlanError::Unsupported("draft head output normalization")),
            };
            if [
                block.embedding_norm.epsilon,
                block.hidden_norm.epsilon,
                output_epsilon,
            ]
            .into_iter()
            .chain(feed_forward.epsilons())
            .any(|epsilon| epsilon != paired.epsilon())
            {
                return Err(PlanError::Unsupported(
                    "normalization epsilons differing within a draft head",
                ));
            }
        }
    }
    draft::admit(definition)
}

/// Admit a head reading target history: every block is a Shared attention
/// followed by a dense feed-forward, under one normalization epsilon.
fn admit_shared_head(head: &SharedHead) -> Result<(), PlanError> {
    for (index, block) in head.blocks.iter().enumerate() {
        let paired = paired_block(block)?;
        head.source(index)?;
        let Some(feed_forward) = paired.feed_forward else {
            return Err(PlanError::Unsupported(
                "draft head block without feed-forward",
            ));
        };
        if !matches!(feed_forward.op, FeedForward::Dense(_)) || paired.per_layer.is_some() {
            return Err(PlanError::Unsupported("draft head feed-forward form"));
        }
        if feed_forward
            .epsilons()
            .into_iter()
            .chain([head.output_norm.epsilon])
            .any(|epsilon| epsilon != paired.epsilon())
        {
            return Err(PlanError::Unsupported(
                "normalization epsilons differing within a draft head",
            ));
        }
    }
    Ok(())
}

/// The attention binding of a block whose mixer must be attention (a draft
/// head's or a separate draft's); `lookup` resolves the planned element of a
/// role in the mixer sublayer's scope, `scalable` that of a projection bound
/// with an accumulator-scale port.
pub(crate) fn attention_slot(
    paired: &PairedBlock,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
    history: magnitude_state::KvCodec,
) -> Result<crate::AttentionBinding, PlanError> {
    let Mixer::Attention(operator) = paired.mixer else {
        return Err(PlanError::Unsupported(
            "a draft block mixer other than attention",
        ));
    };
    if !matches!(operator.reads, HistoryReads::Visible) {
        return Err(PlanError::Unsupported("a draft block's selected attention"));
    }
    let lookup = |kind| paired.mixer_input.lookup(kind, &lookup);
    attention::binding(
        operator,
        paired.mixer_output,
        hidden,
        0,
        lookup,
        scalable,
        activation,
        history,
    )
}

/// The dense binding of a feed-forward that must be dense (a separate
/// draft's); `lookup` resolves the planned element of a role in its scope,
/// `scalable` that of a projection bound with an accumulator-scale port.
pub(crate) fn dense_slot(
    sublayer: &FeedForwardSublayer,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
) -> Result<crate::DenseBinding, PlanError> {
    let FeedForward::Dense(operator) = sublayer.op else {
        return Err(PlanError::Unsupported(
            "a draft feed-forward other than dense",
        ));
    };
    dense_ffn::binding(operator, sublayer.output, lookup, scalable, activation)
}

/// Whether a planned block slot binds the operators of `paired`.
pub(crate) fn slot_matches(paired: &PairedBlock, slot: &crate::TargetBlockProgramSlot) -> bool {
    use crate::{FeedForwardProgramSlot as FeedForwardSlot, MixerProgramSlot as MixerSlot};
    matches!(
        (slot.mixer(), paired.mixer),
        (MixerSlot::Attention(_), Mixer::Attention(_))
            | (MixerSlot::Recurrent(_), Mixer::GatedDelta(_))
            | (MixerSlot::StateSpace(_), Mixer::StateSpace(_))
            | (MixerSlot::ShortConv(_), Mixer::ShortConv(_))
    ) && matches!(
        (
            slot.feed_forward(),
            paired.feed_forward.map(|sublayer| sublayer.op)
        ),
        (None, None)
            | (Some(FeedForwardSlot::Dense(_)), Some(FeedForward::Dense(_)))
            | (
                Some(FeedForwardSlot::Routed(_)),
                Some(FeedForward::Routed(_))
            )
            | (
                Some(FeedForwardSlot::GeneralRouted(_)),
                Some(FeedForward::GeneralRouted(_))
            )
            | (
                Some(FeedForwardSlot::Parallel(_)),
                Some(FeedForward::Parallel(_))
            )
    ) && slot.per_layer().is_some() == paired.per_layer.is_some()
        && slot.ngram().is_some() == paired.ngram.is_some()
}

/// The program slot of a paired block's mixer in a decoder serving
/// `context` tokens; `lookup` resolves the planned element of a role in the
/// mixer sublayer's scope, `scalable` that of a projection bound with an
/// accumulator-scale port.
pub(crate) fn mixer_slot(
    paired: &PairedBlock,
    hidden: u64,
    context: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
    history: magnitude_state::KvCodec,
) -> Result<crate::MixerProgramSlot, PlanError> {
    use crate::MixerProgramSlot as Slot;
    let lookup = |kind| paired.mixer_input.lookup(kind, &lookup);
    Ok(match paired.mixer {
        Mixer::Attention(operator) => Slot::Attention(attention::binding(
            operator,
            paired.mixer_output,
            hidden,
            context,
            lookup,
            &scalable,
            activation,
            history,
        )?),
        Mixer::GatedDelta(operator) => {
            Slot::Recurrent(gated_delta::binding(operator, lookup, &scalable, activation)?)
        }
        Mixer::StateSpace(operator) => {
            Slot::StateSpace(state_space::binding(operator, hidden, lookup, scalable, activation)?)
        }
        Mixer::ShortConv(operator) => {
            Slot::ShortConv(short_conv::binding(operator, hidden, lookup, &scalable, activation)?)
        }
    })
}

/// The program slot of a paired block's feed-forward in `scope`; `lookup`
/// resolves the planned element of a role in a scope, `scalable` that of a
/// projection bound with an accumulator-scale port.
pub(crate) fn feed_forward_slot(
    sublayer: &FeedForwardSublayer,
    hidden: u64,
    scope: WeightScope,
    lookup: impl Fn(WeightScope, WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightScope, WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
) -> Result<crate::FeedForwardProgramSlot, PlanError> {
    use crate::FeedForwardProgramSlot as Slot;
    let local = |kind| match sublayer.input {
        Some(input) => input.lookup(kind, &|kind| lookup(scope, kind)),
        None => lookup(scope, kind),
    };
    let local_scalable = |kind| scalable(scope, kind);
    Ok(match sublayer.op {
        FeedForward::Dense(operator) => Slot::Dense(dense_ffn::binding(
            operator,
            sublayer.output,
            local,
            local_scalable,
            activation,
        )?),
        FeedForward::Routed(operator) => {
            Slot::Routed(routed::fused_binding(operator, hidden, local, local_scalable, activation)?)
        }
        FeedForward::GeneralRouted(operator) => Slot::GeneralRouted(routed::binding(
            operator,
            hidden,
            local,
            local_scalable,
            activation,
        )?),
        FeedForward::Parallel(branches) => {
            let WeightScope::TargetSublayer(index) = scope else {
                return Err(PlanError::Unsupported(
                    "parallel branches outside the target",
                ));
            };
            Slot::Parallel(parallel::binding(
                branches,
                index,
                sublayer.output,
                hidden,
                &lookup,
                &scalable,
                activation,
            )?)
        }
    })
}

/// The mixer and feed-forward of a block, or the typed reason the executor
/// cannot run it.
pub(crate) fn paired_block(block: &Block) -> Result<PairedBlock<'_>, PlanError> {
    let (mixer, feed_forward, trailing) = match block.sublayers.as_slice() {
        [mixer] => (mixer, None, None),
        [mixer, feed_forward] => (mixer, Some(feed_forward), None),
        [mixer, feed_forward, trailing] => (mixer, Some(feed_forward), Some(trailing)),
        _ => {
            return Err(PlanError::Unsupported(
                "block other than one mixer, at most one feed-forward and at most one per-layer \
                 or n-gram input sublayer",
            ));
        }
    };
    let (per_layer, ngram) = match trailing {
        None => (None, None),
        Some(sublayer) if matches!(sublayer.op, Operator::NGramInput(_)) => {
            (None, Some(ngram::admit(sublayer)?))
        }
        Some(sublayer) => (Some(per_layer::admit(sublayer)?), None),
    };
    fn source(input: &InputNorm) -> Result<SublayerSource<'_>, PlanError> {
        match input {
            InputNorm::Rms(norm) => Ok(SublayerSource::Norm(norm)),
            InputNorm::HyperConnectionMix(mix) => Ok(SublayerSource::Mix(mix)),
            _ => Err(PlanError::Unsupported("sublayer input normalization")),
        }
    }
    // The residual joins: a residual add (or a post-norm tail), or the
    // scatter of a mix site's sublayer into the streams.
    let joins = |output: &OutputForm| matches!(output, OutputForm::Residual | OutputForm::Scatter);
    let mixer_input = source(&mixer.input)?;
    let mixer_op = match &mixer.op {
        Operator::Attention(operator) => {
            attention::admit(operator, mixer_input.epsilon())?;
            Mixer::Attention(operator)
        }
        Operator::GatedDelta(operator) => {
            gated_delta::admit(operator, mixer_input.epsilon())?;
            Mixer::GatedDelta(operator)
        }
        Operator::StateSpace(space) => {
            state_space::admit(space)?;
            Mixer::StateSpace(space)
        }
        Operator::ShortConv(conv) => {
            short_conv::admit(conv)?;
            Mixer::ShortConv(conv)
        }
        op => return Err(unsupported_operator(op, "mixer")),
    };
    // Post-norm tails exist for the attention and dense output projections.
    if !matches!(mixer_op, Mixer::Attention(_)) && !joins(&mixer.output) {
        return Err(PlanError::Unsupported(
            "sublayer post-norm after this operator",
        ));
    }
    let feed_forward = feed_forward
        .map(|feed_forward| {
            if let Operator::Parallel(branches) = &feed_forward.op {
                return Ok(FeedForwardSublayer {
                    input: None,
                    op: FeedForward::Parallel(parallel::admit(
                        branches,
                        &feed_forward.input,
                        &feed_forward.output,
                    )?),
                    output: &feed_forward.output,
                });
            }
            let op = match &feed_forward.op {
                Operator::DenseFfn(dense) => {
                    dense_ffn::admit(dense)?;
                    FeedForward::Dense(dense)
                }
                Operator::RoutedFfn(routed) => {
                    routed_feed_forward(routed, source(&feed_forward.input)?.epsilon())?
                }
                op => return Err(unsupported_operator(op, "feed-forward")),
            };
            if !matches!(op, FeedForward::Dense(_)) && !joins(&feed_forward.output) {
                return Err(PlanError::Unsupported(
                    "sublayer post-norm after this operator",
                ));
            }
            Ok(FeedForwardSublayer {
                input: Some(source(&feed_forward.input)?),
                op,
                output: &feed_forward.output,
            })
        })
        .transpose()?;
    Ok(PairedBlock {
        mixer_input,
        mixer: mixer_op,
        mixer_output: &mixer.output,
        feed_forward,
        per_layer,
        ngram,
    })
}

fn unsupported_operator(op: &Operator, position: &'static str) -> PlanError {
    PlanError::UnsupportedOperator {
        operator: op.name(),
        position,
    }
}

/// A routed operator in Qwen's fused form, else the general routed form
/// (`routed::admit`), whose router input norm must share the sublayer's
/// epsilon (`routed_select` takes one).
fn routed_feed_forward<'a>(
    routed: &'a RoutedFfn,
    input_epsilon: f64,
) -> Result<FeedForward<'a>, PlanError> {
    if routed::is_fused(routed) {
        return Ok(FeedForward::Routed(routed));
    }
    routed::admit(routed)?;
    if let RouterInput::Residual(norm) = &routed.router.input {
        if norm.epsilon != input_epsilon {
            return Err(PlanError::Unsupported(
                "expert router norm epsilon differing from its input norm",
            ));
        }
    }
    Ok(FeedForward::GeneralRouted(routed))
}
