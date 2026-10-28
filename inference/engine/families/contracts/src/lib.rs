//! Numerical model-family variation described as data.
//!
//! This crate deliberately contains no loading, execution, batching, state,
//! generation, serving, chat, or template implementation. A family adapter
//! recognizes an artifact and assembles these contracts; common engine crates
//! consume them without depending on that family.

// The closed input contract carries these generic values across the service
// boundary; consumers need not depend on artifact preparation itself.
pub use magnitude_artifacts::{
    ArtifactIdentity, BoundaryRule, InputLayout, InputSpan, PackageIdentity, TokenId,
};
use serde::{Deserialize, Serialize};
use std::{error, fmt};

mod decoder;
mod draft;
mod family;
mod images;
mod inputs;
mod vision;
pub use decoder::{
    ActivationFunction, Attention, AttentionGate, Block, Branch, BranchOutput, Decoder,
    DeferredForm, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection,
    FeedForwardUp, GateFunction, GateGranularity, GatedDelta, HashRoutingTable, HeadNorm,
    HistoryDomain, HistoryReads, HyperConnectionMixing, HyperConnectionWeights, HyperConnections,
    Indexer, InputNorm, KeyValue, LatentAttention, LatentExperts, LayerNorm, MediaRowAttention,
    NGramHash, NGramInput, Operator, OutputForm, PerLayerEntry, PerLayerInput,
    RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary, RotaryDivisors, RotaryPair,
    RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction, SharedExpert,
    SharedExpertGate, ShortConv, StateSpace, Sublayer, SublayerIndex, UnweightedRms, ValueNorm,
    ValueSource,
};
pub use draft::{
    Affine, AffinePart, BlockAttention, BlockConvolutions, BlockLayout, CORRECTOR_FEATURES,
    CandidateCorrector, CandidateSelector, CenteredNorm, ConfidenceHead, CorrectorLayer,
    CorrectorSite, CorrectorWeight, DraftDefinition, DraftEmbedding, DraftMethod, DraftVariant,
    DynamicConvolution, LayerConvolutions, MarkovHead, TapPoint,
};
pub use family::{FamilyError, FamilyInputAdapter, MarkerTokens, ModelFamily};
pub use images::{
    ImageCrop, ImageMarkers, SequentialImageInput, SpatialControls, image_crop_identity,
    image_crops,
};
pub use inputs::{
    InputPreparationError, ModelInputAdapter, PreparedModelInput, PreparedVisionInput, TokenPlan,
    VisionPositionBlend, VisionSpatialControls,
};
pub use vision::{
    CellReduction, LinearClamp, LinearPart, MergerStage, NormPart, PositionSampling, Standardize,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionGatedConvolution, VisionLinear, VisionLinearSite,
    VisionMerger, VisionNorm, VisionNormSite, VisionOperator, VisionPositions, VisionPreprocessing,
    VisionResampling, VisionResize, VisionStem, VisionUp, VisionWeight,
};

/// Stable identity selected by a family adapter after artifact recognition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FamilyId(pub String);

/// Compact activation storage chosen by a model family.
///
/// Runtime-specific dtype objects are intentionally excluded from this
/// contract; an executor translates this value at its implementation seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationDType {
    F16,
    BF16,
}

impl ActivationDType {
    /// Bytes of one stored activation element.
    pub const fn bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// One family-bound tensor role, independent of its container encoding.
///
/// `shape` is the logical shape the engine consumes. When `transforms` is
/// empty it is the stored shape; otherwise the stored tensor becomes the
/// logical one through exact import-time operations, applied in order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeightDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
    pub transforms: Vec<ImportTransform>,
}

impl WeightDescriptor {
    /// A stored tensor consumed as it is.
    pub fn stored(name: impl Into<String>, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            transforms: Vec::new(),
        }
    }

    /// The logical shape `transforms` make of a stored tensor of `stored`
    /// shape.
    pub fn transformed_shape(&self, stored: &[u64]) -> Result<Vec<u64>, DefinitionError> {
        let mut shape = stored.to_vec();
        for transform in &self.transforms {
            let axis = row_axis(&shape).ok_or_else(|| {
                DefinitionError::new(format!("{:?} has no row axis to transform", self.name))
            })?;
            match transform {
                ImportTransform::Rows(range) => {
                    let end = range.start.checked_add(range.rows);
                    if range.rows == 0 || end.is_none_or(|end| end > shape[axis]) {
                        return Err(DefinitionError::new(format!(
                            "row range of {:?} exceeds its rows",
                            self.name
                        )));
                    }
                    shape[axis] = range.rows;
                }
                ImportTransform::PermuteRows { order } => {
                    let period = order.len() as u64;
                    let mut seen = vec![false; order.len()];
                    let bijective = order.iter().all(|&row| {
                        usize::try_from(row)
                            .ok()
                            .and_then(|row| seen.get_mut(row))
                            .is_some_and(|seen| !std::mem::replace(seen, true))
                    });
                    if period == 0 || !bijective || !shape[axis].is_multiple_of(period) {
                        return Err(DefinitionError::new(format!(
                            "row permutation of {:?} is not a bijection over its row groups",
                            self.name
                        )));
                    }
                }
                ImportTransform::Scale { factor } => {
                    if !factor.is_finite() || *factor == 0.0 {
                        return Err(DefinitionError::new(format!(
                            "scale of {:?} is not a finite nonzero factor",
                            self.name
                        )));
                    }
                }
                ImportTransform::Flatten => {
                    shape = vec![checked_product(&shape)?];
                }
                ImportTransform::DropUnitAxis => {
                    if shape.len() < 2 || shape[0] != 1 {
                        return Err(DefinitionError::new(format!(
                            "{:?} has no outer unit axis to drop",
                            self.name
                        )));
                    }
                    shape.remove(0);
                }
                ImportTransform::ScaleByTensor { tensor, .. } => {
                    if tensor.is_empty() || *tensor == self.name {
                        return Err(DefinitionError::new(format!(
                            "scale tensor of {:?} must name another stored tensor",
                            self.name
                        )));
                    }
                }
                ImportTransform::Progressive { plane, depth } => {
                    let [rows, columns] = shape[..] else {
                        return Err(DefinitionError::new(format!(
                            "{:?} is not a matrix of whole 32-value groups",
                            self.name
                        )));
                    };
                    if !columns.is_multiple_of(32) {
                        return Err(DefinitionError::new(format!(
                            "{:?} is not a matrix of whole 32-value groups",
                            self.name
                        )));
                    }
                    shape = plane.shape(rows, columns, *depth);
                }
            }
        }
        Ok(shape)
    }
}

/// The axis import transforms address: a vector's elements, or the rows of
/// a matrix (of each matrix of a stack).
fn row_axis(shape: &[u64]) -> Option<usize> {
    match shape.len() {
        0 => None,
        1 => Some(0),
        rank => Some(rank - 2),
    }
}

/// An exact import-time operation on a stored tensor (model-family plan §2.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImportTransform {
    /// Keep rows `[start, start + rows)`.
    Rows(RowRange),
    /// Within every consecutive group of `order.len()` rows, logical row `i`
    /// is stored row `order[i]`.
    PermuteRows { order: Vec<u64> },
    /// Multiply every element by `factor` (dense encodings only).
    Scale { factor: f64 },
    /// Read the stored tensor as one vector of all its elements in storage
    /// order (a parameter vector stored as `[1, n]` or `[groups, n/groups]`).
    Flatten,
    /// Read the stored tensor without its outermost axis, of extent one (a
    /// matrix stored with a trailing unit GGML dimension).
    DropUnitAxis,
    /// Multiply every matrix of a stack (the whole tensor when it is not a
    /// stack) by the matching element of the stored F32 tensor `tensor`,
    /// shaped `[1]` or `[stack]`. A container format's second-level scale
    /// (NVFP4) stays a separate factor of the resident weight, so it is exact
    /// for block encodings too. `activations` is how the weight's producer
    /// calibrated its products' activations.
    ScaleByTensor {
        tensor: String,
        activations: ScaledActivations,
    },
    /// One plane of a symmetric 32-value code matrix (the last
    /// transform; `ProgressivePlane`).
    Progressive {
        plane: ProgressivePlane,
        depth: ProgressiveCodeDepth,
    },
}

/// The exact symmetric 32-value code representation a progressive placement
/// preserves. The coarse view owns half its bits, then one refinement bit;
/// the remaining planes complete the original codes without duplicated weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProgressiveCodeDepth {
    Four,
    Eight,
}

impl ProgressiveCodeDepth {
    pub const fn bits(self) -> u8 {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
    pub const fn coarse_bits(self) -> u8 {
        self.bits() / 2
    }
}

/// The activations a second-level-scaled (NVFP4) weight's products take, as
/// its producer calibrated them: NVIDIA's `W4A16_NVFP4` stores the weight
/// and its scale, its `NVFP4` (W4A4) also the activation scale
/// `<stem>.input_scale` (docs/precision.md, "NVFP4 operands").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaledActivations {
    /// The activation as it is rounded (W4A16).
    Rounded,
    /// The activation's rows as NVFP4 operands (W4A4).
    Nvfp4,
}

/// The progressive placement of a matrix `[rows, columns]`: its
/// offset-binary codes `u = c + 2^(depth-1)` in significance-ordered bit planes, so a
/// reader of the top bits reads only them (`readout.seismic`, the progressive
/// head readout). The planes hold the matrix's values bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProgressivePlane {
    /// The coarse half of each code, packed into u32 words.
    Top,
    /// The next refinement bit, a u32 per 32-value group.
    Next,
    /// The remaining significance-ordered bit planes of each group.
    Rest,
    /// The f16 scale of every group (`[rows, columns / 32]`).
    Scales,
    /// Per row, bounds on how far a dot product moves from the coarse and refined
    /// views to the exact row, per unit Euclidean length of the activation
    /// row, F32 rounding included (`[2, rows]` F32).
    Radius,
}

impl ProgressivePlane {
    pub const ALL: [Self; 5] = [
        Self::Top,
        Self::Next,
        Self::Rest,
        Self::Scales,
        Self::Radius,
    ];

    /// The plane's shape for a `[rows, columns]` matrix.
    pub fn shape(self, rows: u64, columns: u64, depth: ProgressiveCodeDepth) -> Vec<u64> {
        match self {
            Self::Top => vec![rows, columns / 32, u64::from(depth.coarse_bits())],
            Self::Next | Self::Scales => vec![rows, columns / 32],
            Self::Rest => vec![
                rows,
                columns / 32,
                u64::from(depth.bits() - depth.coarse_bits() - 1),
            ],
            Self::Radius => vec![rows, 2],
        }
    }
}

// A transform's factor is finite and nonzero (`transformed_shape` rejects
// anything else), so equality is reflexive, agrees with bit equality, and
// descriptors compare and hash as values.
impl Eq for ImportTransform {}

impl std::hash::Hash for ImportTransform {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Rows(range) => (range.start, range.rows).hash(state),
            Self::PermuteRows { order } => order.hash(state),
            Self::Scale { factor } => factor.to_bits().hash(state),
            Self::Flatten | Self::DropUnitAxis => {}
            Self::ScaleByTensor {
                tensor,
                activations,
            } => {
                tensor.hash(state);
                activations.hash(state);
            }
            Self::Progressive { plane, depth } => {
                plane.hash(state);
                depth.hash(state);
            }
        }
    }
}

/// Structural location of a numerical weight. This identity is independent of
/// the model family's container name and therefore remains stable after family
/// metadata has been interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightScope {
    /// Entry and exit weights of the decoder.
    Target,
    TargetSublayer(SublayerIndex),
    TargetBranch {
        sublayer: SublayerIndex,
        branch: u32,
    },
    /// Weights of a draft head block outside its decoder blocks' sublayers.
    HeadBlock(u32),
    /// A sublayer of a draft head's decoder block (`block` counts the
    /// decoder blocks of every head block in order).
    HeadSublayer(SublayerIndex),
    HeadBranch {
        sublayer: SublayerIndex,
        branch: u32,
    },
    /// Projector weights outside its patch frames, blocks and merger stages.
    Vision,
    /// One frame of a projector's patch convolution.
    VisionPatch(u32),
    VisionBlock(u32),
    VisionMerger(u32),
    /// Weights of a separate draft component outside its layers.
    Draft,
    /// A sublayer of a draft component's layers.
    DraftSublayer(SublayerIndex),
    /// One transformer layer of a draft's candidate corrector.
    CorrectorLayer(u32),
}

/// Closed semantic role within a structural weight scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightKind {
    Embedding,
    PerLayerTable,
    PerLayerModelProjection,
    PerLayerProjectionNorm,
    HashRoutingTable,
    /// A sublayer's or branch's input normalization.
    InputNorm,
    /// A sublayer's or branch's output normalization.
    PostNorm,
    LayerScale,
    /// A hyper-connection site's weights: a sublayer's mix (sublayer
    /// scope) or the exit's (target scope).
    HyperConnectionNorm,
    HyperConnectionDown,
    HyperConnectionUp,
    HyperConnectionInject,
    Query,
    QueryGate,
    AttentionGate,
    Key,
    Value,
    QueryNorm,
    KeyNorm,
    AttentionOutput,
    /// An attention's per-head sink logits (`Attention::sinks`).
    AttentionSinks,
    IndexerQuery,
    IndexerKey,
    IndexerQueryNorm,
    IndexerKeyNorm,
    RecurrentQueryKeyValue,
    RecurrentGate,
    RecurrentAlpha,
    RecurrentBeta,
    RecurrentConvolution,
    RecurrentConvolutionBias,
    RecurrentDecay,
    RecurrentTimeBias,
    RecurrentNorm,
    RecurrentOutput,
    ShortConvInputGate,
    ShortConvValue,
    ShortConvOutputGate,
    StateSpaceProjection,
    StateSpaceSkip,
    /// The grouped norm of a state-space output (F32 in the kernels, unlike
    /// the gated delta norm).
    StateSpaceNorm,
    DenseGate,
    DenseUp,
    DenseDown,
    Router,
    RouterInputNorm,
    RouterSelectionBias,
    SharedRouter,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    ExpertScale,
    LatentDown,
    LatentUp,
    SharedGate,
    SharedUp,
    SharedDown,
    PerLayerGate,
    PerLayerProjection,
    NGramTable,
    NGramKey,
    NGramValue,
    NGramKeyNorm,
    NGramQueryNorm,
    NGramConvolutionNorm,
    NGramConvolution,
    RotaryDivisors,
    OutputNorm,
    Output,
    /// The model-defined centroid projection for clustered proposals.
    OutputCentroids,
    /// A plane of the output projection's progressive placement
    /// (`ImportTransform::Progressive`), which then replaces `Output`.
    OutputPlane(ProgressivePlane),
    HeadEmbeddingNorm,
    HeadHiddenNorm,
    HeadCombine,
    /// A draft head's projection of `[embedding ‖ hidden]` into its width
    /// (`HeadInput::Projected`).
    HeadInputProjection,
    /// A draft head's projection of its normalized output into the hidden
    /// row its next chained step conditions on (`HeadChain::Projected`).
    HeadChainProjection,
    /// A draft's projection of the concatenated target taps.
    DraftFusion,
    DraftFusionNorm,
    MarkovEmbedding,
    MarkovProjection,
    ConfidenceWeight,
    ConfidenceBias,
    /// A convolved draft sublayer's convolution base `[2, kernel, hidden]`
    /// and coefficient projection (scope `DraftSublayer`).
    ConvolutionBase,
    ConvolutionProjection,
    /// DFlash2's candidate selector: hidden projection and codebooks.
    SelectorHidden,
    SelectorPredecessor,
    SelectorSuccessor,
    /// A LiLiCorr candidate corrector weight (scopes `Draft` and
    /// `CorrectorLayer`).
    Corrector(CorrectorWeight),
    /// A projector weight (scopes `Vision`, `VisionPatch`, `VisionBlock`,
    /// `VisionMerger`).
    Vision(VisionWeight),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeightRole {
    pub scope: WeightScope,
    pub kind: WeightKind,
}

/// Numerically relevant input rules supplied by a family adapter. Token
/// coordinates themselves are part of each prepared input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSemantics {
    pub coordinate_axes: u8,
}

impl InputSemantics {
    pub fn validate(&self) -> Result<(), DefinitionError> {
        if self.coordinate_axes == 0 {
            return Err(DefinitionError::new(
                "model coordinates require at least one axis",
            ));
        }
        Ok(())
    }
}

pub fn checked_product(dimensions: &[u64]) -> Result<u64, DefinitionError> {
    dimensions
        .iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| DefinitionError::new("model dimensions overflow"))
}

/// How a draft head forms its input row from the target's embedding `e` of
/// the next token and a conditioning hidden row `h` (the target's normalized
/// output row, or the head's own chain feature).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HeadInput {
    /// `combine · [RMS(e)·w_e ‖ RMS(h)·w_h]`, at the target's width.
    Combine {
        embedding_norm: RmsNorm,
        hidden_norm: RmsNorm,
        /// `[hidden, 2·hidden]`.
        combine: WeightDescriptor,
    },
    /// `pre · [e·scale ‖ h]`, into the head's own width.
    Projected {
        /// `[width, 2·hidden]`.
        pre: WeightDescriptor,
        embedding_scale: EmbeddingScale,
    },
}

/// The vocabulary projection, independent of proposal candidate policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum VocabularyProjection {
    /// The target's output projection.
    Target,
    /// The head's own table, `[vocabulary, width]`.
    Own { table: WeightDescriptor },
}

/// Model-defined candidate policy for a proposal distribution.
/// This never changes the target's exact readout or acceptance semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProposalReadoutPolicy {
    Full,
    Clustered {
        /// `[clusters, width]` centroid projection.
        centroids: WeightDescriptor,
        /// Cluster-major permutation of every canonical vocabulary token.
        token_order: Vec<TokenId>,
        selected_clusters: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadReadout {
    pub projection: VocabularyProjection,
    pub policy: ProposalReadoutPolicy,
}

impl HeadReadout {
    pub fn full(projection: VocabularyProjection) -> Self {
        Self {
            projection,
            policy: ProposalReadoutPolicy::Full,
        }
    }

    pub fn own_table(&self) -> Option<&WeightDescriptor> {
        match &self.projection {
            VocabularyProjection::Target => None,
            VocabularyProjection::Own { table } => Some(table),
        }
    }
}

/// The hidden row a chained step conditions on after the step before it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HeadChain {
    /// The head's normalized output.
    Output,
    /// The head's residual streams before its output normalization/mix.
    Residual,
    /// `post ·` the head's normalized output: `[hidden, width]`.
    Projected { post: WeightDescriptor },
}

/// The attention history a draft head's blocks read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadHistory {
    /// The head stores its own rows: every committed row enters it, and its
    /// blocks' attention owns or shares history among themselves.
    Own,
    /// The head stores nothing. Every attention sublayer is
    /// `KeyValue::Shared` with a source naming a sublayer of the *target*
    /// decoder, whose accepted history it reads; a draft is one row per step
    /// at the anchor's position.
    Target,
}

/// The package component a draft head's weights are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadComponent {
    /// Embedded in the target artifact.
    Target,
    /// The package's separate draft component.
    Draft,
}

/// One draft head following the target decoder: it forms its input from the
/// embedding of the next token and a hidden row, runs its decoder blocks at
/// its own width and normalizes their output for its vocabulary projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadBlock {
    pub residual: ResidualForm,
    pub input: HeadInput,
    pub blocks: Vec<Block>,
    pub output_norm: ExitNorm,
    pub readout: HeadReadout,
    pub chain: HeadChain,
}

impl HeadBlock {
    /// The residual width of the head's blocks.
    pub fn width(&self, decoder: &Decoder) -> u64 {
        match &self.input {
            HeadInput::Combine { .. } => decoder.hidden,
            HeadInput::Projected { pre, .. } => pre.shape.first().copied().unwrap_or(0),
        }
    }
}

/// A draft head (MTP): chained blocks that read the target's embedding and a
/// hidden row and propose the tokens after the anchor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Head {
    pub blocks: Vec<HeadBlock>,
    pub history: HeadHistory,
    pub component: HeadComponent,
}

impl Head {
    /// A residual-chaining head consumes the target's residual streams; an
    /// output-chaining head consumes its normalized output feature.
    pub fn residual_conditioning(&self) -> bool {
        self.blocks
            .first()
            .is_some_and(|block| matches!(block.chain, HeadChain::Residual))
    }

    pub fn conditioning_width(&self, decoder: &Decoder) -> u64 {
        if self.residual_conditioning() {
            match &decoder.residual {
                ResidualForm::Single => decoder.hidden,
                ResidualForm::HyperConnections(form) => decoder.hidden * form.streams,
            }
        } else {
            decoder.hidden
        }
    }
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    pub fn validate(&self, decoder: &Decoder, coordinate_axes: u8) -> Result<(), DefinitionError> {
        if self.blocks.is_empty() {
            return Err(DefinitionError::new("draft head has no blocks"));
        }
        let combined_hidden = checked_product(&[2, decoder.hidden])?;
        for block in &self.blocks {
            if matches!(block.chain, HeadChain::Residual) != self.residual_conditioning() {
                return Err(DefinitionError::new(
                    "head blocks must preserve their conditioning representation",
                ));
            }
            let width = block.width(decoder);
            match &block.input {
                HeadInput::Combine {
                    embedding_norm,
                    hidden_norm,
                    combine,
                } => {
                    decoder::validate_rms(embedding_norm, decoder.hidden, "draft head norm")?;
                    decoder::validate_rms(
                        hidden_norm,
                        self.conditioning_width(decoder),
                        "draft head norm",
                    )?;
                    expect_shape(
                        combine,
                        &[decoder.hidden, combined_hidden],
                        "draft head input projection",
                    )?;
                }
                HeadInput::Projected { pre, .. } => {
                    if width == 0 {
                        return Err(DefinitionError::new("draft head has no width"));
                    }
                    expect_shape(
                        pre,
                        &[width, combined_hidden],
                        "draft head input projection",
                    )?;
                }
            }
            if block.blocks.is_empty() {
                return Err(DefinitionError::new("draft head has no decoder blocks"));
            }
            let context = decoder::Context {
                hidden: width,
                vocabulary: decoder.vocabulary,
                coordinate_axes,
                residual: &block.residual,
                per_layer: None,
                shared_sources: match self.history {
                    HeadHistory::Own => None,
                    HeadHistory::Target => Some(&decoder.blocks),
                },
            };
            decoder::validate_blocks(&block.blocks, &context)?;
            decoder::validate_exit_norm(&block.output_norm, &context, "draft head norm")?;
            if let Some(table) = block.readout.own_table() {
                expect_shape(
                    table,
                    &[decoder.vocabulary, width],
                    "draft head readout table",
                )?;
            }
            if let ProposalReadoutPolicy::Clustered {
                centroids,
                token_order,
                selected_clusters,
                ..
            } = &block.readout.policy
            {
                let clusters = centroids.shape.first().copied().unwrap_or(0);
                if clusters == 0
                    || *selected_clusters == 0
                    || *selected_clusters > clusters
                    || decoder.vocabulary % clusters != 0
                {
                    return Err(DefinitionError::new(
                        "invalid proposal readout cluster geometry",
                    ));
                }
                expect_shape(centroids, &[clusters, width], "proposal readout centroids")?;
                if token_order.len() as u64 != decoder.vocabulary {
                    return Err(DefinitionError::new(
                        "proposal readout ordering must cover the vocabulary",
                    ));
                }
                let mut seen = vec![false; token_order.len()];
                for token in token_order {
                    let Some(entry) = seen.get_mut(token.0 as usize) else {
                        return Err(DefinitionError::new(
                            "proposal readout ordering has a noncanonical token",
                        ));
                    };
                    if std::mem::replace(entry, true) {
                        return Err(DefinitionError::new(
                            "proposal readout ordering repeats a token",
                        ));
                    }
                }
            }
            match &block.chain {
                HeadChain::Residual
                    if block.residual != decoder.residual || width != decoder.hidden =>
                {
                    return Err(DefinitionError::new(
                        "a residual-chaining head must preserve the target's residual form",
                    ));
                }
                HeadChain::Residual => {}
                HeadChain::Output if width != decoder.hidden => {
                    return Err(DefinitionError::new(
                        "a draft head of its own width chains through a projection",
                    ));
                }
                HeadChain::Output => {}
                HeadChain::Projected { post } => {
                    expect_shape(
                        post,
                        &[decoder.hidden, width],
                        "draft head chain projection",
                    )?;
                }
            }
            if matches!(block.readout.projection, VocabularyProjection::Target)
                && width != decoder.hidden
            {
                return Err(DefinitionError::new(
                    "a draft head of its own width reads out through its own table",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowRange {
    pub start: u64,
    pub rows: u64,
}

/// Shared numerical description assembled by a model-family adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub family: FamilyId,
    pub artifact_identity: PackageIdentity,
    pub inputs: InputSemantics,
    pub decoder: Decoder,
    pub head: Option<Head>,
    pub vision: Option<VisionDescription>,
    /// The package's separate draft component.
    pub draft: Option<DraftDefinition>,
}

impl ModelDefinition {
    pub fn validate(&self) -> Result<(), DefinitionError> {
        if self.family.0.is_empty() {
            return Err(DefinitionError::new("model family identity is empty"));
        }
        self.inputs.validate()?;
        self.decoder.validate(self.inputs.coordinate_axes)?;
        if let Some(head) = &self.head {
            head.validate(&self.decoder, self.inputs.coordinate_axes)?;
        }
        if let Some(vision) = &self.vision {
            vision.validate()?;
            if vision.output_hidden != self.decoder.hidden {
                return Err(DefinitionError::new(
                    "image features must have the decoder's width",
                ));
            }
        }
        if let Some(draft) = &self.draft {
            draft.validate(&self.decoder, self.inputs.coordinate_axes)?;
        }
        Ok(())
    }

    /// The deferred forms (plan §9) this definition uses, in model order.
    /// An executor rejects a definition that uses any.
    pub fn deferred_forms(&self) -> Vec<DeferredForm> {
        let mut forms = Vec::new();
        self.decoder.deferred_forms(&mut forms);
        if let Some(draft) = &self.draft {
            decoder::blocks_deferred_forms(&draft.blocks, &mut forms);
        }
        if let Some(head) = &self.head {
            for block in &head.blocks {
                decoder::blocks_deferred_forms(&block.blocks, &mut forms);
            }
        }
        forms
    }
}

pub(crate) fn expect_shape(
    descriptor: &WeightDescriptor,
    expected: &[u64],
    role: &str,
) -> Result<(), DefinitionError> {
    if descriptor.shape != expected {
        return Err(DefinitionError::new(format!(
            "invalid {role} shape for {:?}",
            descriptor.name
        )));
    }
    checked_product(expected)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefinitionError(String);

impl DefinitionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for DefinitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for DefinitionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::ArtifactIdentity;

    #[test]
    fn direct_dependencies_preserve_the_family_contract_boundary() {
        let manifest = include_str!("../Cargo.toml");
        let forbidden = [
            "seismic",
            "magnitude-executor",
            "magnitude-generation",
            "magnitude-scheduler",
            "magnitude-chat",
            "magnitude-templates",
        ];
        for dependency in forbidden {
            assert!(
                !manifest.lines().any(|line| {
                    let line = line.trim_start();
                    line.starts_with(dependency) && line[dependency.len()..].starts_with([' ', '='])
                }),
                "family contracts must not depend on {dependency}"
            );
        }
    }

    fn weight(name: &str, shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor::stored(name, shape)
    }

    fn rms(name: &str, width: u64) -> RmsNorm {
        RmsNorm {
            weight: weight(name, &[width]),
            epsilon: 1e-5,
        }
    }

    fn attention(name: &str, key_value: KeyValue) -> Sublayer {
        Sublayer {
            input: InputNorm::Rms(rms(&format!("{name}.norm"), 4)),
            op: Operator::Attention(Box::new(Attention {
                heads: 2,
                kv_heads: 1,
                width: 2,
                query: weight(&format!("{name}.q"), &[4, 4]),
                gate: AttentionGate::None,
                query_norm: HeadNorm::None,
                key_value,
                rotary: Rotary::Table {
                    pairs: vec![RotaryPair {
                        frequency: 1.0,
                        amplitude: 1.0,
                    }],
                    divisors: None,
                },
                scale: 0.5,
                reads: HistoryReads::Visible,
                media_rows: MediaRowAttention::Causal,
                sinks: None,
                output: weight(&format!("{name}.o"), &[4, 4]),
            })),
            output: OutputForm::Residual,
        }
    }

    fn owned(name: &str, domain: HistoryDomain) -> KeyValue {
        KeyValue::Owned {
            key: weight(&format!("{name}.k"), &[2, 4]),
            value: ValueSource::Projected(weight(&format!("{name}.v"), &[2, 4])),
            key_norm: HeadNorm::None,
            value_norm: ValueNorm::None,
            domain,
        }
    }

    fn definition(sublayers: Vec<Sublayer>) -> ModelDefinition {
        ModelDefinition {
            family: FamilyId("fixture".into()),
            artifact_identity: PackageIdentity {
                target: ArtifactIdentity([0; 32]),
                projector: None,
            },
            inputs: InputSemantics { coordinate_axes: 1 },
            decoder: Decoder {
                activation_dtype: ActivationDType::BF16,
                hidden: 4,
                vocabulary: 8,
                context_limit: 16,
                residual: ResidualForm::Single,
                entry: EntryForm {
                    embedding: weight("embedding", &[8, 4]),
                    scale: EmbeddingScale::Unit,
                    norm: None,
                    per_layer: None,
                    hash_routing: None,
                },
                blocks: sublayers
                    .into_iter()
                    .map(|sublayer| Block {
                        sublayers: vec![sublayer],
                    })
                    .collect(),
                exit: ExitForm {
                    norm: ExitNorm::Rms(rms("output_norm", 4)),
                    output: weight("embedding", &[8, 4]),
                    softcap: None,
                },
            },
            head: None,
            vision: None,
            draft: None,
        }
    }

    #[test]
    fn operators_bind_weights_of_their_own_geometry() {
        let mut model = definition(vec![attention("a", owned("a", HistoryDomain::Token))]);
        assert!(model.validate().is_ok());
        let Operator::Attention(attention) = &mut model.decoder.blocks[0].sublayers[0].op else {
            unreachable!()
        };
        attention.gate = AttentionGate::Interleaved {
            function: GateFunction::Sigmoid,
        };
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "invalid attention query shape for \"a.q\""
        );
    }

    #[test]
    fn shared_history_must_name_an_earlier_owning_attention_layer() {
        let source = SublayerIndex {
            block: 0,
            sublayer: 0,
        };
        let model = definition(vec![
            attention("a", owned("a", HistoryDomain::Window { tokens: 4 })),
            attention("b", KeyValue::Shared { source }),
        ]);
        assert!(model.validate().is_ok());
        let backwards = definition(vec![
            attention(
                "a",
                KeyValue::Shared {
                    source: SublayerIndex {
                        block: 1,
                        sublayer: 0,
                    },
                },
            ),
            attention("b", owned("b", HistoryDomain::Token)),
        ]);
        assert_eq!(
            backwards.validate().unwrap_err().to_string(),
            "shared attention history names no earlier attention sublayer"
        );
    }

    #[test]
    fn a_head_reading_target_history_names_target_layers_and_owns_none() {
        let target_layer = SublayerIndex {
            block: 0,
            sublayer: 0,
        };
        let head = |key_value, history| Head {
            blocks: vec![HeadBlock {
                residual: ResidualForm::Single,
                input: HeadInput::Projected {
                    pre: weight("pre", &[4, 8]),
                    embedding_scale: EmbeddingScale::Unit,
                },
                blocks: vec![Block {
                    sublayers: vec![attention("h", key_value)],
                }],
                output_norm: ExitNorm::Rms(rms("head_norm", 4)),
                readout: HeadReadout::full(VocabularyProjection::Own {
                    table: weight("head_table", &[8, 4]),
                }),
                chain: HeadChain::Projected {
                    post: weight("post", &[4, 4]),
                },
            }],
            history,
            component: HeadComponent::Draft,
        };
        let mut model = definition(vec![attention("a", owned("a", HistoryDomain::Token))]);
        model.head = Some(head(
            KeyValue::Shared {
                source: target_layer,
            },
            HeadHistory::Target,
        ));
        assert!(model.validate().is_ok());
        model.head = Some(head(owned("h", HistoryDomain::Token), HeadHistory::Target));
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "attention reading target history owns its own"
        );
        model.head = Some(head(
            KeyValue::Shared {
                source: SublayerIndex {
                    block: 1,
                    sublayer: 0,
                },
            },
            HeadHistory::Target,
        ));
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "shared attention history names no target attention sublayer"
        );
        // The same source names the head's own layers when it keeps its own.
        model.head = Some(head(
            KeyValue::Shared {
                source: target_layer,
            },
            HeadHistory::Own,
        ));
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "shared attention history names no earlier attention sublayer"
        );
    }

    #[test]
    fn deferred_forms_are_named_not_rejected_by_the_contract() {
        let model = definition(vec![attention(
            "a",
            owned("a", HistoryDomain::Block { rate: 4 }),
        )]);
        assert!(model.validate().is_ok());
        assert_eq!(model.deferred_forms(), [DeferredForm::BlockHistory]);
    }

    /// A two-stream, rank-3 site over hidden 4.
    fn site(name: &str, inject: bool) -> HyperConnectionWeights {
        HyperConnectionWeights {
            norm: rms(&format!("{name}.hc_norm"), 8),
            down: weight(&format!("{name}.hc_down"), &[3, 8]),
            up: weight(&format!("{name}.hc_up"), &[8, 3]),
            inject: inject.then(|| weight(&format!("{name}.hc_inject"), &[2, 8])),
        }
    }

    fn ngram() -> Sublayer {
        Sublayer {
            input: InputNorm::None,
            op: Operator::NGramInput(Box::new(NGramInput {
                hash: NGramHash {
                    context: 2,
                    heads_per_order: 1,
                    multipliers: vec![3, 5, 7],
                    sizes: vec![5, 7],
                    offsets: vec![0, 5],
                    reset: 7,
                },
                width: 2,
                table: weight("ple.table", &[12, 2]),
                key: weight("ple.key", &[8, 4]),
                value: weight("ple.value", &[4, 4]),
                key_norm: rms("ple.norm_key", 8),
                query_norm: rms("ple.norm_query", 8),
                convolution_norm: rms("ple.norm_conv", 8),
                convolution: weight("ple.conv", &[8, 4]),
                kernel: 4,
                dilation: 3,
            })),
            output: OutputForm::Residual,
        }
    }

    fn hyper_connected(sublayers: Vec<Sublayer>) -> ModelDefinition {
        let mut model = definition(sublayers);
        model.decoder.residual = ResidualForm::HyperConnections(HyperConnections {
            streams: 2,
            mixing: HyperConnectionMixing::LowRankGated { rank: 3 },
        });
        model.decoder.exit.norm = ExitNorm::HyperConnectionHead(site("output", false));
        model
    }

    fn mixed(mut sublayer: Sublayer, name: &str) -> Sublayer {
        sublayer.input = InputNorm::HyperConnectionMix(site(name, true));
        sublayer.output = OutputForm::Scatter;
        sublayer
    }

    #[test]
    fn hyper_connections_with_selected_history_and_ngram_input_validate() {
        let mut selected = attention("a", owned("a", HistoryDomain::Token));
        let Operator::Attention(operator) = &mut selected.op else {
            unreachable!()
        };
        operator.reads = HistoryReads::Selected {
            block: 4,
            blocks: 2,
            indexer: Indexer {
                heads: 2,
                width: 2,
                query: weight("a.indexer.q", &[4, 4]),
                key: weight("a.indexer.k", &[2, 4]),
                query_norm: rms("a.indexer.q_norm", 2),
                key_norm: rms("a.indexer.k_norm", 2),
            },
        };
        let model = hyper_connected(vec![ngram(), mixed(selected, "a")]);
        model.validate().unwrap();
        assert_eq!(model.deferred_forms(), []);
    }

    #[test]
    fn sublayers_must_match_the_residual_form() {
        let plain = attention("a", owned("a", HistoryDomain::Token));
        let rejects = |model: ModelDefinition| model.validate().unwrap_err().to_string();
        assert_eq!(
            rejects(hyper_connected(vec![plain.clone()])),
            "attention sublayer disagrees with the residual form"
        );
        assert_eq!(
            rejects(definition(vec![mixed(plain.clone(), "a")])),
            "attention sublayer disagrees with the residual form"
        );
        assert_eq!(
            rejects(definition(vec![ngram()])),
            "n-gram input sublayer disagrees with the residual form"
        );
        let mut exit_injects = hyper_connected(vec![mixed(plain, "a")]);
        exit_injects.decoder.exit.norm = ExitNorm::HyperConnectionHead(site("output", true));
        assert_eq!(
            rejects(exit_injects),
            "decoder output norm disagrees with the residual form"
        );
    }

    #[test]
    fn ngram_products_must_be_exact() {
        let mut sublayer = ngram();
        let Operator::NGramInput(input) = &mut sublayer.op else {
            unreachable!()
        };
        input.hash.multipliers[2] = u64::MAX / 4;
        assert_eq!(
            hyper_connected(vec![sublayer])
                .validate()
                .unwrap_err()
                .to_string(),
            "invalid n-gram input geometry"
        );
    }

    /// The reference's vectorized rule (`references/families/qwen4exp.py`
    /// `ngram_rows`, mlx's `ngram_ids`): the sequence after its `context`
    /// previous tokens (reset tokens before the sequence), each shift valid
    /// while the position's segment since the latest reset strictly before
    /// it reaches back that far.
    fn reference_rows(hash: &NGramHash, tokens: &[u64]) -> Vec<u64> {
        let previous = vec![hash.reset; hash.context as usize];
        reference_rows_after(hash, &previous, tokens)
    }

    fn reference_rows_after(hash: &NGramHash, previous: &[u64], tokens: &[u64]) -> Vec<u64> {
        let context = hash.context as usize;
        assert_eq!(previous.len(), context);
        let history = previous.iter().chain(tokens).copied().collect::<Vec<_>>();
        let mut rows = Vec::new();
        for at in context..history.len() {
            let previous = history[..at]
                .iter()
                .rposition(|token| *token == hash.reset)
                .map_or(-1, |index| index as i64);
            let in_segment = at as i64 - (previous + 1);
            let shifted = (0..=context)
                .map(|shift| match in_segment >= shift as i64 {
                    true => history[at - shift],
                    false => hash.reset,
                })
                .collect::<Vec<_>>();
            for order in 2..=context + 1 {
                let mixed = (1..order).fold(shifted[0] * hash.multipliers[0], |mixed, distance| {
                    mixed ^ (shifted[distance] * hash.multipliers[distance])
                });
                let first = (order - 2) * hash.heads_per_order as usize;
                for head in first..first + hash.heads_per_order as usize {
                    rows.push(mixed % hash.sizes[head] + hash.offsets[head]);
                }
            }
        }
        rows
    }

    #[test]
    fn ngram_rows_follow_the_reference_across_chunks() {
        let hash = NGramHash {
            context: 2,
            heads_per_order: 2,
            multipliers: vec![1_000_003, 999_983, 1_000_033],
            sizes: vec![101, 103, 107, 109],
            offsets: vec![0, 101, 204, 311],
            reset: 7,
        };
        // Resets at the start, adjacent and mid-sequence.
        let mut state = 11u64;
        let tokens = (0..64)
            .map(|index| match index {
                0 | 9 | 10 | 33 => 7,
                _ => {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 33) % 50
                }
            })
            .collect::<Vec<_>>();
        let whole = hash.rows(&[], &tokens);
        assert_eq!(whole, reference_rows(&hash, &tokens));
        // Chunked: each chunk reads the tokens before it.
        for split in [1, 2, 9, 10, 11, 34, 63] {
            let mut chunked = hash.rows(&[], &tokens[..split]);
            chunked.extend(hash.rows(&tokens[..split], &tokens[split..]));
            assert_eq!(chunked, whole, "split at {split}");
        }
    }

    /// mlx's `ngram_ids` cases (`test_integer_planner_matches_library_eos_and_batch`):
    /// every third token the reset, after a previous context that holds a
    /// reset and a token, at lengths 1, 2, 19 and 512; multipliers of the
    /// released size (just under `i64::MAX / vocabulary`).
    #[test]
    fn ngram_rows_follow_the_reference_after_a_previous_context() {
        let hash = NGramHash {
            context: 2,
            heads_per_order: 8,
            multipliers: vec![
                (i64::MAX as u64 / 248_320) | 1,
                (i64::MAX as u64 / 248_320 / 3) | 1,
                (i64::MAX as u64 / 248_320 / 7) | 1,
            ],
            sizes: (0..16).map(|head| 20_000_003 + 2 * head).collect(),
            offsets: (0..16).map(|head| head * 20_000_040).collect(),
            reset: 2,
        };
        for length in [1usize, 2, 19, 512] {
            let mut state = length as u64;
            let tokens = (0..length)
                .map(|index| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match index % 3 {
                        0 => 2,
                        _ => (state >> 33) % 100,
                    }
                })
                .collect::<Vec<_>>();
            for previous in [[2, 7], [8, 2]] {
                assert_eq!(
                    hash.rows(&previous, &tokens),
                    reference_rows_after(&hash, &previous, &tokens),
                    "length {length}, previous {previous:?}"
                );
            }
        }
    }

    #[test]
    fn import_transforms_derive_the_logical_shape() {
        let split = WeightDescriptor {
            name: "in_proj".into(),
            shape: vec![4, 8],
            transforms: vec![
                ImportTransform::Rows(RowRange { start: 4, rows: 4 }),
                ImportTransform::PermuteRows {
                    order: vec![0, 2, 1, 3],
                },
            ],
        };
        assert_eq!(split.transformed_shape(&[12, 8]).unwrap(), [4, 8]);
        assert!(split.transformed_shape(&[6, 8]).is_err());
        let not_bijective = WeightDescriptor {
            transforms: vec![ImportTransform::PermuteRows { order: vec![0, 0] }],
            ..split
        };
        assert!(not_bijective.transformed_shape(&[4, 8]).is_err());
        let flattened = WeightDescriptor {
            name: "ssm_norm".into(),
            shape: vec![8],
            transforms: vec![ImportTransform::Flatten],
        };
        assert_eq!(flattened.transformed_shape(&[2, 4]).unwrap(), [8]);
        let scaled = WeightDescriptor {
            name: "experts".into(),
            shape: vec![2, 4, 8],
            transforms: vec![ImportTransform::ScaleByTensor {
                tensor: "experts.scale".into(),
                activations: ScaledActivations::Rounded,
            }],
        };
        assert_eq!(scaled.transformed_shape(&[2, 4, 8]).unwrap(), [2, 4, 8]);
    }
}
