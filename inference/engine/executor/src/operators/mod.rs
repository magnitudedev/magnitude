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
pub(crate) mod output;
pub(crate) mod parallel;
pub(crate) mod per_layer;
pub(crate) mod routed;
pub(crate) mod short_conv;
pub(crate) mod state_space;
pub(crate) mod vision;

use crate::error::PlanError;
use crate::SublayerTail;
use magnitude_family_contracts::{ShortConv, StateSpace};
use magnitude_family_contracts::{
    Attention, Block, BranchOutput, Decoder, DenseFfn, ExitNorm, GatedDelta, Head, InputNorm,
    ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm, RoutedFfn, RouterInput,
    Sublayer, SublayerIndex, WeightDescriptor, WeightKind, WeightRole, WeightScope,
};
use seismic::Element;

/// An operator module's sink for the weights it binds, by semantic kind.
pub(super) type WeightPush<'p, 'a> = dyn FnMut(WeightKind, &'a WeightDescriptor) + 'p;

/// The tail of a sublayer with `output`; `lookup` resolves the planned
/// element of a role in the sublayer's scope.
pub(crate) fn tail(
    output: &OutputForm,
    lookup: &impl Fn(WeightKind) -> Result<Element, PlanError>,
) -> Result<SublayerTail, PlanError> {
    match output {
        OutputForm::Residual => Ok(SublayerTail::Residual),
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

/// Every weight of the decoder's sublayers with its semantic role, in model
/// order. Entry and exit weights are the caller's (`WeightScope::Target`).
pub(crate) fn decoder_weights(decoder: &Decoder) -> Vec<(WeightRole, &WeightDescriptor)> {
    let mut weights = Vec::new();
    for (index, sublayer) in decoder.sublayers() {
        sublayer_weights(sublayer, WeightScope::TargetSublayer(index), &mut weights, |branch| {
            WeightScope::TargetBranch {
                sublayer: index,
                branch,
            }
        });
    }
    weights
}

/// Every weight of a draft head, in model order.
pub(crate) fn head_weights(head: &Head) -> Result<Vec<(WeightRole, &WeightDescriptor)>, PlanError> {
    let mut weights = Vec::new();
    for (block, head_block) in head.blocks.iter().enumerate() {
        let block = u32::try_from(block).map_err(|_| PlanError::Arithmetic("head block index"))?;
        let scope = WeightScope::HeadBlock(block);
        let role = |kind| WeightRole { scope, kind };
        weights.push((role(WeightKind::HeadEmbeddingNorm), &head_block.embedding_norm.weight));
        weights.push((role(WeightKind::HeadHiddenNorm), &head_block.hidden_norm.weight));
        weights.push((role(WeightKind::HeadCombine), &head_block.combine));
        for (sublayer, value) in head_block.block.sublayers.iter().enumerate() {
            let index = SublayerIndex {
                block,
                sublayer: u32::try_from(sublayer)
                    .map_err(|_| PlanError::Arithmetic("head sublayer index"))?,
            };
            sublayer_weights(value, WeightScope::HeadSublayer(index), &mut weights, |branch| {
                WeightScope::HeadBranch {
                    sublayer: index,
                    branch,
                }
            });
        }
        weights.push((role(WeightKind::OutputNorm), head_block.output_norm.weight()));
    }
    Ok(weights)
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
        OutputForm::Residual => {}
        OutputForm::PostNorm(norm) => weights.push((role(scope, WeightKind::PostNorm), &norm.weight)),
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
        InputNorm::HyperConnectionMix(mix) => {
            weights.push((role(scope, WeightKind::HyperConnectionMix), &mix.mix))
        }
        InputNorm::RmsUnweighted(_) | InputNorm::None => {}
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
            | WeightKind::RecurrentConvolution
            | WeightKind::RecurrentConvolutionBias
            | WeightKind::RecurrentDecay
            | WeightKind::RecurrentTimeBias
            | WeightKind::StateSpaceSkip
            | WeightKind::StateSpaceNorm
            | WeightKind::RouterSelectionBias
            | WeightKind::ExpertScale
            | WeightKind::SharedRouter
            | WeightKind::ConfidenceWeight
            | WeightKind::ConfidenceBias
            | WeightKind::ConvolutionBase
    )
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
    pub mixer_norm: &'a RmsNorm,
    pub mixer: Mixer<'a>,
    pub mixer_output: &'a OutputForm,
    pub feed_forward: Option<FeedForwardSublayer<'a>>,
    /// The per-layer input sublayer after the feed-forward (Gemma E2B/E4B):
    /// unnormalized input, a post-norm tail.
    pub per_layer: Option<per_layer::PerLayerSublayer<'a>>,
}

/// The feed-forward sublayer of a [`PairedBlock`]. Its input norm is absent
/// only for parallel branches, which normalize their own inputs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FeedForwardSublayer<'a> {
    pub norm: Option<&'a RmsNorm>,
    pub op: FeedForward<'a>,
    pub output: &'a OutputForm,
}

impl FeedForwardSublayer<'_> {
    /// The epsilons of every norm the sublayer's kernels share one epsilon
    /// with: its input norm, or its branches' input and output norms.
    pub fn epsilons(&self) -> Vec<f64> {
        match self.op {
            FeedForward::Parallel(branches) => {
                branches.norms().iter().map(|norm| norm.epsilon).collect()
            }
            _ => self.norm.iter().map(|norm| norm.epsilon).collect(),
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
            Self::GatedDelta(_) | Self::StateSpace(_) | Self::ShortConv(_) => {
                MixerKind::Recurrent
            }
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
/// components of every earlier recurrent mixer come first.
pub(crate) fn bank_component_index(blocks: &[Block], index: usize) -> Result<usize, PlanError> {
    blocks[..index].iter().try_fold(0, |components, block| {
        Ok(components + paired_block(block)?.mixer.bank_components())
    })
}

impl PairedBlock<'_> {
    /// The one normalization epsilon the block's kernels take.
    pub fn epsilon(&self) -> f64 {
        self.mixer_norm.epsilon
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
        Ok(format!(
            "{mixer} {:?}; {feed_forward:?}; {per_layer:?}; {}",
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
        OutputForm::Residual => None,
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
    if decoder.residual != ResidualForm::Single {
        return Err(PlanError::Unsupported("residual form"));
    }
    // `embedding_rows` scales (`embedding_transform`) and normalizes text
    // rows. Media rows enter as their projector leaves them: a family whose
    // text rows are normalized at entry normalizes its media rows in its
    // projector (Muse's `perception_emb_norm`).
    attention::admit_media(decoder, definition.vision.is_some())?;
    let ExitNorm::Rms(exit) = &decoder.exit.norm else {
        return Err(PlanError::Unsupported("readout normalization"));
    };
    for block in &decoder.blocks {
        let paired = paired_block(block)?;
        if paired
            .feed_forward
            .is_some_and(|feed_forward| {
                feed_forward
                    .epsilons()
                    .iter()
                    .any(|epsilon| *epsilon != paired.epsilon())
            })
            || exit.epsilon != paired.epsilon()
        {
            return Err(PlanError::Unsupported(
                "normalization epsilons differing within a decoder",
            ));
        }
    }
    if let Some(head) = definition.head.as_ref().filter(|_| head) {
        for block in &head.blocks {
            let paired = paired_block(&block.block)?;
            if !matches!(paired.mixer, Mixer::Attention(_)) {
                return Err(PlanError::Unsupported("draft head mixer"));
            }
            let Some(feed_forward) = paired.feed_forward else {
                return Err(PlanError::Unsupported("draft head block without feed-forward"));
            };
            if matches!(
                feed_forward.op,
                FeedForward::GeneralRouted(_) | FeedForward::Parallel(_)
            ) {
                return Err(PlanError::Unsupported("draft head routed feed-forward form"));
            }
            if [paired.mixer_output, feed_forward.output]
                .iter()
                .any(|output| **output != OutputForm::Residual)
            {
                return Err(PlanError::Unsupported("draft head sublayer output form"));
            }
            let ExitNorm::Rms(output) = &block.output_norm else {
                return Err(PlanError::Unsupported("draft head output normalization"));
            };
            if [
                block.embedding_norm.epsilon,
                block.hidden_norm.epsilon,
                output.epsilon,
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

/// The attention binding of a block whose mixer must be attention (a draft
/// head's or a separate draft's); `lookup` resolves the planned element of a
/// role in the mixer sublayer's scope.
pub(crate) fn attention_slot(
    paired: &PairedBlock,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
    history: magnitude_state::KvCodec,
) -> Result<crate::AttentionBinding, PlanError> {
    let Mixer::Attention(operator) = paired.mixer else {
        return Err(PlanError::Unsupported("a draft block mixer other than attention"));
    };
    attention::binding(operator, paired.mixer_output, hidden, lookup, activation, history)
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
        return Err(PlanError::Unsupported("a draft feed-forward other than dense"));
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
        (slot.feed_forward(), paired.feed_forward.map(|sublayer| sublayer.op)),
        (None, None)
            | (Some(FeedForwardSlot::Dense(_)), Some(FeedForward::Dense(_)))
            | (Some(FeedForwardSlot::Routed(_)), Some(FeedForward::Routed(_)))
            | (Some(FeedForwardSlot::GeneralRouted(_)), Some(FeedForward::GeneralRouted(_)))
            | (Some(FeedForwardSlot::Parallel(_)), Some(FeedForward::Parallel(_)))
    ) && slot.per_layer().is_some() == paired.per_layer.is_some()
}

/// The program slot of a paired block's mixer; `lookup` resolves the planned
/// element of a role in the mixer sublayer's scope.
pub(crate) fn mixer_slot(
    paired: &PairedBlock,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
    history: magnitude_state::KvCodec,
) -> Result<crate::MixerProgramSlot, PlanError> {
    use crate::MixerProgramSlot as Slot;
    Ok(match paired.mixer {
        Mixer::Attention(operator) => Slot::Attention(attention::binding(
            operator,
            paired.mixer_output,
            hidden,
            lookup,
            activation,
            history,
        )?),
        Mixer::GatedDelta(operator) => {
            Slot::Recurrent(gated_delta::binding(operator, lookup, activation)?)
        }
        Mixer::StateSpace(operator) => {
            Slot::StateSpace(state_space::binding(operator, hidden, lookup, activation)?)
        }
        Mixer::ShortConv(operator) => {
            Slot::ShortConv(short_conv::binding(operator, hidden, lookup, activation)?)
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
    let local = |kind| lookup(scope, kind);
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
            Slot::Routed(routed::fused_binding(operator, hidden, local, activation)?)
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
                return Err(PlanError::Unsupported("parallel branches outside the target"));
            };
            Slot::Parallel(parallel::binding(
                branches,
                index,
                sublayer.output,
                hidden,
                &lookup,
                activation,
            )?)
        }
    })
}

/// The mixer and feed-forward of a block, or the typed reason the executor
/// cannot run it.
pub(crate) fn paired_block(block: &Block) -> Result<PairedBlock<'_>, PlanError> {
    let (mixer, feed_forward, per_layer) = match block.sublayers.as_slice() {
        [mixer] => (mixer, None, None),
        [mixer, feed_forward] => (mixer, Some(feed_forward), None),
        [mixer, feed_forward, per_layer] => (mixer, Some(feed_forward), Some(per_layer)),
        _ => {
            return Err(PlanError::Unsupported(
                "block other than one mixer, at most one feed-forward and at most one per-layer \
                 input sublayer",
            ))
        }
    };
    let per_layer = per_layer.map(per_layer::admit).transpose()?;
    fn norm(input: &InputNorm) -> Result<&RmsNorm, PlanError> {
        match input {
            InputNorm::Rms(norm) => Ok(norm),
            _ => Err(PlanError::Unsupported("sublayer input normalization")),
        }
    }
    let mixer_norm = norm(&mixer.input)?;
    let mixer_op = match &mixer.op {
        Operator::Attention(operator) => {
            attention::admit(operator, mixer_norm.epsilon)?;
            Mixer::Attention(operator)
        }
        Operator::GatedDelta(operator) => {
            gated_delta::admit(operator, mixer_norm.epsilon)?;
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
    if !matches!(mixer_op, Mixer::Attention(_)) && mixer.output != OutputForm::Residual {
        return Err(PlanError::Unsupported("sublayer post-norm after this operator"));
    }
    let feed_forward = feed_forward
        .map(|feed_forward| {
            if let Operator::Parallel(branches) = &feed_forward.op {
                return Ok(FeedForwardSublayer {
                    norm: None,
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
                    routed_feed_forward(routed, norm(&feed_forward.input)?.epsilon)?
                }
                op => return Err(unsupported_operator(op, "feed-forward")),
            };
            if !matches!(op, FeedForward::Dense(_)) && feed_forward.output != OutputForm::Residual
            {
                return Err(PlanError::Unsupported("sublayer post-norm after this operator"));
            }
            Ok(FeedForwardSublayer {
                norm: Some(norm(&feed_forward.input)?),
                op,
                output: &feed_forward.output,
            })
        })
        .transpose()?;
    Ok(PairedBlock {
        mixer_norm,
        mixer: mixer_op,
        mixer_output: &mixer.output,
        feed_forward,
        per_layer,
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
