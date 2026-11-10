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
    ExitSite, FeedForwardUp, GateFunction, GluLimit, GateGranularity, GatedDelta, HeadNorm, HistoryDomain,
    BlockElement, BlockRounding, BlockScale, Candidates, HeadWeights, HistoryReads, HistoryRows,
    HistorySource, HyperConnectionExit, HyperConnectionInput, HyperConnectionMixing,
    HyperConnectionWeights, HyperConnections, LowRankGatedSite, Indexer, IndexerInput,
    IndexerKeys, IndexerQuery, InputNorm, LatentAttention, Pool, PooledRows, ProjectedRows,
    RowProducer, RowRounding, Selection, TokenDomain, LatentExperts, LayerNorm, MediaRowAttention,
    NGramConvolution, NGramGate, NGramHash, NGramInput, NGramMedia, Operator, OutputForm, OutputProjection, OutputRotary, PerLayerEntry,
    PerLayerInput, QueryProjection, RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary,
    RotaryDivisors, RotaryLayout, RotaryPair, RotaryPairing, RotaryPlacement,
    RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction, SharedExpert,
    SharedExpertGate, ShortConv, SinkhornSite, StateSpace, Sublayer, SublayerIndex,
    UnweightedRms, ValueNorm,
    ValueSource, grouped_output_columns,
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

    /// The progressive plane and code depth the descriptor places: its
    /// sole transform's, if that is `ImportTransform::Progressive`.
    pub fn progressive(&self) -> Option<(ProgressivePlane, ProgressiveCodeDepth)> {
        match self.transforms[..] {
            [ImportTransform::Progressive { plane, depth }] => Some((plane, depth)),
            _ => None,
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
                ImportTransform::ScaleBlocks { tensor, tile, .. } => {
                    if tensor.is_empty() || *tensor == self.name {
                        return Err(DefinitionError::new(format!(
                            "scale grid of {:?} must name another stored tensor",
                            self.name
                        )));
                    }
                    if !ImportTransform::tiles(&shape, *tile) {
                        return Err(DefinitionError::new(format!(
                            "{:?} {shape:?} is no matrix, or stack of matrices, of whole \
                             power-of-two {tile:?} tiles",
                            self.name
                        )));
                    }
                }
                ImportTransform::Progressive { plane, depth } => {
                    let whole = || {
                        DefinitionError::new(format!(
                            "{:?} is not a matrix of whole {}-value groups",
                            self.name,
                            ProgressivePlane::GROUP
                        ))
                    };
                    let [rows, columns] = shape[..] else {
                        return Err(whole());
                    };
                    if !columns.is_multiple_of(ProgressivePlane::GROUP) {
                        return Err(whole());
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
    /// Read the stored matrix, or stack of matrices, (of single-value
    /// blocks) under the stored grid `tensor` of block scales: one scale of
    /// `codes` per `tile` (`[rows, columns]`, [`Self::tiles`]) of each
    /// matrix, the grid shaped `[rows / tile[0], columns / tile[1]]` (with
    /// the stack's leading extent). The importer places each tile's scale
    /// beside each block of its rows' values, which replicates it exactly.
    ScaleBlocks {
        tensor: String,
        tile: [u64; 2],
        codes: ScaleCodes,
    },
    /// One plane of a symmetric 32-value code matrix (the last
    /// transform; `ProgressivePlane`).
    Progressive {
        plane: ProgressivePlane,
        depth: ProgressiveCodeDepth,
    },
}

impl ImportTransform {
    /// The stored companion scale of weight `<stem>.weight`: `<stem>.scale`
    /// (`ScaleByTensor`'s tensor, or `ScaleBlocks`' grid).
    pub fn companion_scale(weight: &str) -> Option<String> {
        weight
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.scale"))
    }

    /// The stored grid of block scales (`ScaleBlocks`) a weight `weight` of
    /// `encoding` is read under: the companion scale
    /// ([`Self::companion_scale`]) of an F8_E4M3 weight, whose blocks carry
    /// no scale of their own; `None` for a weight read under its own blocks'
    /// scales. The one rule the binder binds a grid by and weight planning
    /// classifies a component's tensors by.
    pub fn scale_grid(
        weight: &str,
        encoding: magnitude_artifacts::gguf::Encoding,
    ) -> Option<String> {
        match encoding {
            magnitude_artifacts::gguf::Encoding::F8E4M3 => Self::companion_scale(weight),
            _ => None,
        }
    }

    /// The tile a stored grid `grid` of block scales gives a `shape` matrix
    /// or stack of matrices, when the grid tiles it ([`Self::tiles`]):
    /// `[rows / g, columns / h]` for a `[g, h]` grid (`[matrices, g, h]` of a
    /// `[matrices, rows, columns]` stack, the same tile for every matrix)
    /// dividing it exactly.
    pub fn grid_tile(shape: &[u64], grid: &[u64]) -> Option<[u64; 2]> {
        let (Some(([rows, columns], stack)), Some(([grid_rows, grid_columns], grid_stack))) =
            (Self::matrices(shape), Self::matrices(grid))
        else {
            return None;
        };
        if stack != grid_stack
            || grid_rows == 0
            || grid_columns == 0
            || !rows.is_multiple_of(grid_rows)
            || !columns.is_multiple_of(grid_columns)
        {
            return None;
        }
        let tile = [rows / grid_rows, columns / grid_columns];
        Self::tiles(shape, tile).then_some(tile)
    }

    /// Whether `tile` tiles every matrix of a `shape` matrix or stack of
    /// matrices exactly, each extent a power of two dividing the matrix's.
    ///
    /// A file stores no tile, only its grid, so the tile is inferred from
    /// the shapes, and this rule is what makes the inferred tile the true
    /// one. Producers tile by powers of two, `T`, with `g = ceil(n / T)`
    /// tiles over an extent `n`; the inferred `t = n / g` is admitted only
    /// when `g` divides `n` and `t` is a power of two. If `t < T`, then
    /// `t <= T / 2`, so `n = g t <= g T / 2`, while `n > (g - 1) T`: that
    /// holds only for `g = 1`, where every row (column) is in the one tile
    /// under either. If `t > T`, then `t >= 2 T` and `n = g t >= 2 g T > n`.
    /// So `t = T` whenever the extent has more than one tile, and a grid
    /// over a partial last tile (`300 = 10 x 30`) is refused, not misread.
    pub fn tiles(shape: &[u64], tile: [u64; 2]) -> bool {
        let Some(([rows, columns], _)) = Self::matrices(shape) else {
            return false;
        };
        tile[0].is_power_of_two()
            && tile[1].is_power_of_two()
            && rows.is_multiple_of(tile[0])
            && columns.is_multiple_of(tile[1])
    }

    /// The `[rows, columns]` of a matrix (`[rows, columns]`) or of each
    /// matrix of a stack (`[matrices, rows, columns]`), with the stack's
    /// matrices (`None` for a matrix); `None` for any other rank.
    fn matrices(shape: &[u64]) -> Option<([u64; 2], Option<u64>)> {
        match *shape {
            [rows, columns] => Some(([rows, columns], None)),
            [stack, rows, columns] => Some(([rows, columns], Some(stack))),
            _ => None,
        }
    }
}

/// The codes of a stored block-scale grid (`ImportTransform::ScaleBlocks`),
/// named by the type the grid is stored as ([`Self::of_stored`]): a
/// block's value is its code times its tile's scale, one F32 product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaleCodes {
    /// UE8M0 bytes, `2^(b - 127)`, 0xff NaN (OCP E8M0), stored as I8.
    Ue8m0,
    /// F32 values, stored as F32: an FP8 checkpoint's `weight_scale_inv`,
    /// each finite, and zero or of magnitude at least 2^-117, which the
    /// importer checks.
    F32,
}

impl ScaleCodes {
    pub const ALL: [Self; 2] = [Self::Ue8m0, Self::F32];

    /// The artifact encoding a grid of them is stored as.
    pub const fn stored(self) -> magnitude_artifacts::gguf::Encoding {
        match self {
            Self::Ue8m0 => magnitude_artifacts::gguf::Encoding::I8,
            Self::F32 => magnitude_artifacts::gguf::Encoding::F32,
        }
    }

    /// The codes of a grid stored as `encoding`, if it is any codes' type:
    /// each codes' stored type is theirs alone, so the type names them.
    pub fn of_stored(encoding: magnitude_artifacts::gguf::Encoding) -> Option<Self> {
        Self::ALL.into_iter().find(|codes| codes.stored() == encoding)
    }
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
    /// The bit planes of `ProgressivePlane::Rest`: the bits below the coarse
    /// ones and the next.
    pub const fn rest_planes(self) -> u8 {
        self.bits() - self.coarse_bits() - 1
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

    /// Values per group: the planes hold one scale, and one word of each bit
    /// plane, per group.
    pub const GROUP: u64 = 32;

    /// The plane's shape for a `[rows, columns]` matrix.
    pub fn shape(self, rows: u64, columns: u64, depth: ProgressiveCodeDepth) -> Vec<u64> {
        let groups = columns / Self::GROUP;
        match self {
            Self::Top => vec![rows, groups, u64::from(depth.coarse_bits())],
            Self::Next | Self::Scales => vec![rows, groups],
            Self::Rest => vec![rows, groups, u64::from(depth.rest_planes())],
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
            Self::ScaleBlocks {
                tensor,
                tile,
                codes,
            } => {
                tensor.hash(state);
                tile.hash(state);
                codes.hash(state);
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
    /// A Sinkhorn site's projection, bias and scales.
    HyperConnectionFunction,
    HyperConnectionBase,
    HyperConnectionScale,
    /// The query heads' projection: a full query's, or a low-rank query's
    /// `up` (`QueryProjection`).
    Query,
    QueryGate,
    /// A low-rank query's `down` projection and the norm of its `qr` row.
    QueryDown,
    QueryDownNorm,
    AttentionGate,
    Key,
    Value,
    QueryNorm,
    KeyNorm,
    /// The projection to the hidden width: a full output's, or a grouped
    /// output's `b` (`OutputProjection`).
    AttentionOutput,
    /// A grouped output's block-diagonal `a`.
    AttentionOutputGrouped,
    /// An attention's per-head sink logits (`Attention::sinks`).
    AttentionSinks,
    /// A pooled history source's per-token projection, gate and norm
    /// (`PooledRows`).
    PooledProjection,
    PooledGate,
    PooledNorm,
    IndexerQuery,
    /// An indexer's key projection: its pooled keys' projection, or its
    /// latent keys' (`IndexerKeys`).
    IndexerKey,
    /// The gate of an indexer's softmax-gated pooled keys.
    IndexerKeyGate,
    IndexerQueryNorm,
    IndexerKeyNorm,
    /// An indexer's projected per-head score weights (`HeadWeights`).
    IndexerHeadWeights,
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
    /// The n-gram gate's product weight (`NGramGate::Product`).
    NGramGate,
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
    /// The head stores nothing. Every attention source is
    /// `HistoryRows::Shared` with a source naming a sublayer of the *target*
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

    /// A `GluLimit` is positive and finite at F32, and a definition's limit
    /// deserializes only as one; it serializes as the plain number, and its
    /// kernels' LB is L's F32 bits, 0 for none.
    #[test]
    fn a_glu_limit_is_positive_and_finite_at_f32() {
        for invalid in [0.0, -0.0, -10.0, 1e-50, 1e39, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(GluLimit::new(invalid).is_err(), "{invalid} admitted");
        }
        for invalid in ["0.0", "-10.0", "1e-50", "1e39"] {
            assert!(serde_json::from_str::<GluLimit>(invalid).is_err(), "{invalid} deserialized");
        }
        let limit = GluLimit::new(10.0).unwrap();
        assert_eq!(serde_json::to_string(&Some(limit)).unwrap(), "10.0");
        assert_eq!(serde_json::to_string(&None::<GluLimit>).unwrap(), "null");
        assert_eq!(serde_json::from_str::<Option<GluLimit>>("10.0").unwrap(), Some(limit));
        assert_eq!((f64::from(limit), GluLimit::word(Some(limit))), (10.0, u64::from(10.0f32.to_bits())));
        // No limit's word is 0, the one word no limit takes.
        assert_eq!(GluLimit::word(None), 0);
    }

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

    fn attention(name: &str, history: Vec<HistorySource>) -> Sublayer {
        Sublayer {
            input: InputNorm::Rms(rms(&format!("{name}.norm"), 4)),
            op: Operator::Attention(Box::new(Attention {
                heads: 2,
                kv_heads: 1,
                width: 2,
                query: QueryProjection::Full(weight(&format!("{name}.q"), &[4, 4])),
                gate: AttentionGate::None,
                query_norm: HeadNorm::None,
                history,
                rotary: Rotary::Table {
                    pairs: vec![RotaryPair {
                        frequency: 1.0,
                        amplitude: 1.0,
                    }],
                    layout: RotaryLayout::HALF_SPLIT_LEADING,
                    divisors: None,
                    output: OutputRotary::None,
                },
                scale: 0.5,
                media_rows: MediaRowAttention::Causal,
                sinks: None,
                output: OutputProjection::Full(weight(&format!("{name}.o"), &[4, 4])),
            })),
            output: OutputForm::Residual,
        }
    }

    fn projected(name: &str, domain: TokenDomain) -> ProjectedRows {
        ProjectedRows {
            key: weight(&format!("{name}.k"), &[2, 4]),
            value: ValueSource::Projected(weight(&format!("{name}.v"), &[2, 4])),
            key_norm: HeadNorm::None,
            value_norm: ValueNorm::None,
            domain,
            rounding: RowRounding::None,
        }
    }

    /// One owned projected source read whole.
    fn owned(name: &str, domain: TokenDomain) -> Vec<HistorySource> {
        vec![HistorySource::visible(HistoryRows::Own(
            RowProducer::Projected(projected(name, domain)),
        ))]
    }

    /// The first source of sublayer `source`, read whole.
    fn shared(source: SublayerIndex) -> Vec<HistorySource> {
        vec![HistorySource::visible(HistoryRows::Shared { source, part: 0 })]
    }

    fn attention_of(model: &mut ModelDefinition, block: usize) -> &mut Attention {
        let Operator::Attention(attention) = &mut model.decoder.blocks[block].sublayers[0].op
        else {
            unreachable!()
        };
        attention
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
        let mut model = definition(vec![attention("a", owned("a", TokenDomain::Token))]);
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

    /// DeepSeek V4.1's window attention: a low-rank query with its own norm,
    /// values that are the stored key, adjacent pairs over the trailing
    /// dimensions, the rotation undone on the output, a grouped low-rank
    /// output, sinks.
    #[test]
    fn low_rank_stored_key_trailing_adjacent_grouped_attention_validates() {
        let mut model = definition(vec![attention(
            "a",
            vec![HistorySource::visible(HistoryRows::Own(RowProducer::Projected(
                stored_key_window("a"),
            )))],
        )]);
        let Operator::Attention(attention) = &mut model.decoder.blocks[0].sublayers[0].op else {
            unreachable!()
        };
        attention.query = QueryProjection::LowRank {
            down: weight("a.q_a", &[3, 4]),
            norm: rms("a.q_a_norm", 3),
            up: weight("a.q_b", &[4, 3]),
        };
        let Rotary::Table { layout, output, .. } = &mut attention.rotary else {
            unreachable!()
        };
        *layout = RotaryLayout {
            pairs: RotaryPairing::Adjacent,
            placement: RotaryPlacement::Trailing,
        };
        *output = OutputRotary::Inverse;
        attention.sinks = Some(weight("a.sinks", &[2]));
        attention.output = OutputProjection::GroupedLowRank {
            groups: 2,
            rank: 3,
            a: weight("a.o_a", &[6, 2]),
            b: weight("a.o_b", &[4, 6]),
        };
        model.validate().unwrap();

        let invalid = |change: &dyn Fn(&mut Attention), error: &str| {
            let mut model = model.clone();
            let Operator::Attention(attention) = &mut model.decoder.blocks[0].sublayers[0].op
            else {
                unreachable!()
            };
            change(attention);
            assert_eq!(model.validate().unwrap_err().to_string(), error);
        };
        invalid(
            &|attention| {
                let HistoryRows::Own(RowProducer::Projected(rows)) =
                    &mut attention.history[0].rows
                else {
                    unreachable!()
                };
                rows.value_norm = ValueNorm::RmsUnweighted(UnweightedRms { epsilon: 1e-6 });
            },
            "stored-key values with a value norm",
        );
        invalid(
            &|attention| {
                let OutputProjection::GroupedLowRank { groups, .. } = &mut attention.output else {
                    unreachable!()
                };
                *groups = 3;
            },
            "invalid grouped attention output",
        );
        invalid(
            &|attention| {
                let QueryProjection::LowRank { up, .. } = &mut attention.query else {
                    unreachable!()
                };
                *up = weight("a.q_b", &[4, 2]);
            },
            "invalid attention query up shape for \"a.q_b\"",
        );
        // Raw-key values are unrotated: an inverse output rotary would turn
        // them away from the rows.
        invalid(
            &|attention| {
                let HistoryRows::Own(RowProducer::Projected(rows)) =
                    &mut attention.history[0].rows
                else {
                    unreachable!()
                };
                rows.value = ValueSource::RawKey;
                rows.rounding = RowRounding::None;
            },
            "an inverse output rotary over values other than the stored keys",
        );
    }

    #[test]
    fn shared_history_must_name_an_earlier_owning_attention_layer() {
        let source = SublayerIndex {
            block: 0,
            sublayer: 0,
        };
        let model = definition(vec![
            attention("a", owned("a", TokenDomain::Window { tokens: 4 })),
            attention("b", shared(source)),
        ]);
        assert!(model.validate().is_ok());
        let backwards = definition(vec![
            attention(
                "a",
                shared(SublayerIndex {
                    block: 1,
                    sublayer: 0,
                }),
            ),
            attention("b", owned("b", TokenDomain::Token)),
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
        let head = |sources, history| Head {
            blocks: vec![HeadBlock {
                residual: ResidualForm::Single,
                input: HeadInput::Projected {
                    pre: weight("pre", &[4, 8]),
                    embedding_scale: EmbeddingScale::Unit,
                },
                blocks: vec![Block {
                    sublayers: vec![attention("h", sources)],
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
        let mut model = definition(vec![attention("a", owned("a", TokenDomain::Token))]);
        model.head = Some(head(shared(target_layer), HeadHistory::Target));
        assert!(model.validate().is_ok());
        model.head = Some(head(owned("h", TokenDomain::Token), HeadHistory::Target));
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "attention reading target history owns its own"
        );
        model.head = Some(head(
            shared(SublayerIndex {
                block: 1,
                sublayer: 0,
            }),
            HeadHistory::Target,
        ));
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "shared attention history names no target attention sublayer"
        );
        // The same source names the head's own layers when it keeps its own.
        model.head = Some(head(shared(target_layer), HeadHistory::Own));
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "shared attention history names no earlier attention sublayer"
        );
    }

    /// V4.1's window rows: the stored key, normed, rounded to E4M3 per 2.
    fn stored_key_window(name: &str) -> ProjectedRows {
        ProjectedRows {
            key: weight(&format!("{name}.kv"), &[2, 4]),
            value: ValueSource::StoredKey,
            key_norm: HeadNorm::Rms(rms(&format!("{name}.kv_norm"), 2)),
            value_norm: ValueNorm::None,
            domain: TokenDomain::Window { tokens: 4 },
            rounding: RowRounding::Block(BlockRounding {
                element: BlockElement::E4M3,
                block: 2,
                scale: BlockScale::Ue8m0,
                amax_floor: 1e-4,
            }),
        }
    }

    fn window_source(name: &str) -> HistorySource {
        HistorySource::visible(HistoryRows::Own(RowProducer::Projected(stored_key_window(
            name,
        ))))
    }

    /// A V4.1 Full layer: its window, then rows pooled per 2 tokens by a
    /// softmax gate and selected by its own indexer over their latents.
    fn compressing(name: &str) -> Sublayer {
        let indexer = Indexer {
            heads: 2,
            width: 2,
            query: IndexerQuery {
                input: IndexerInput::QueryRank,
                projection: weight(&format!("{name}.indexer.q_b"), &[4, 3]),
                norm: HeadNorm::None,
                rounding: RowRounding::None,
            },
            keys: IndexerKeys::Latent {
                projection: weight(&format!("{name}.indexer.k"), &[2, 2]),
                norm: rms(&format!("{name}.indexer.k_norm"), 2),
                rounding: RowRounding::None,
            },
            head_weights: HeadWeights::Projected(weight(&format!("{name}.indexer.w"), &[2, 4])),
            scale: 0.5,
            top: 512,
            candidates: crate::Candidates::All,
        };
        let pooled = HistorySource {
            rows: HistoryRows::Own(RowProducer::Pooled(PooledRows {
                rate: 2,
                projection: weight(&format!("{name}.compressor.kv"), &[2, 4]),
                pool: Pool::SoftmaxGated {
                    gate: weight(&format!("{name}.compressor.gate"), &[2, 4]),
                },
                norm: rms(&format!("{name}.compressor.norm"), 2),
                rounding: RowRounding::Block(BlockRounding {
                    element: BlockElement::E2M1,
                    block: 2,
                    scale: BlockScale::E4M3,
                    amax_floor: 6.0 * 2f64.powi(-9),
                }),
            })),
            reads: HistoryReads::Selected(Selection::Own(Box::new(indexer))),
        };
        low_rank_query(attention(name, vec![window_source(name), pooled]), name)
    }

    /// A V4.1 Reuse layer of block 0's pooled rows and picks.
    fn reusing(name: &str) -> Sublayer {
        let source = SublayerIndex {
            block: 0,
            sublayer: 0,
        };
        let pooled = HistorySource {
            rows: HistoryRows::Shared { source, part: 1 },
            reads: HistoryReads::Selected(Selection::Reuse { source, part: 1 }),
        };
        low_rank_query(attention(name, vec![window_source(name), pooled]), name)
    }

    fn low_rank_query(mut sublayer: Sublayer, name: &str) -> Sublayer {
        let Operator::Attention(attention) = &mut sublayer.op else {
            unreachable!()
        };
        attention.query = QueryProjection::LowRank {
            down: weight(&format!("{name}.q_a"), &[3, 4]),
            norm: rms(&format!("{name}.q_a_norm"), 3),
            up: weight(&format!("{name}.q_b"), &[4, 3]),
        };
        sublayer
    }

    #[test]
    fn compressed_full_and_reuse_layers_validate() {
        let model = definition(vec![compressing("a"), reusing("b")]);
        model.validate().unwrap();
        assert_eq!(model.deferred_forms(), []);
        let invalid = |change: &dyn Fn(&mut ModelDefinition), error: &str| {
            let mut model = model.clone();
            change(&mut model);
            assert_eq!(model.validate().unwrap_err().to_string(), error);
        };
        invalid(
            &|model| attention_of(model, 0).history.swap(0, 1),
            "a pooled first history source",
        );
        invalid(
            &|model| {
                let window = attention_of(model, 0).history[0].clone();
                attention_of(model, 0).history[1] = window;
            },
            "a projected history source after the first",
        );
        invalid(
            &|model| {
                let HistoryRows::Own(RowProducer::Pooled(rows)) =
                    &mut attention_of(model, 0).history[1].rows
                else {
                    unreachable!()
                };
                rows.rate = 1;
            },
            "attention pooled rows: pool and rate disagree",
        );
        invalid(
            &|model| {
                let HistoryRows::Own(RowProducer::Projected(rows)) =
                    &mut attention_of(model, 0).history[0].rows
                else {
                    unreachable!()
                };
                rows.value = ValueSource::RawKey;
            },
            "a rounded history row whose value is not its key",
        );
        invalid(
            &|model| {
                attention_of(model, 1).history[1].reads =
                    HistoryReads::Selected(Selection::Reuse {
                        source: SublayerIndex {
                            block: 0,
                            sublayer: 0,
                        },
                        part: 0,
                    });
            },
            "reused selection names no earlier indexer over the same rows",
        );
        invalid(
            &|model| attention_of(model, 0).query = QueryProjection::Full(weight("a.q", &[4, 4])),
            "indexer reading the query rank of a full-rank query",
        );
    }

    /// A V4.1 Reindex layer: block 0's pooled rows and index keys, its own
    /// query and head weights, restricted to block 0's candidate pool.
    fn reindexing(name: &str) -> Sublayer {
        let source = SublayerIndex {
            block: 0,
            sublayer: 0,
        };
        let mut sublayer = compressing(name);
        let Operator::Attention(attention) = &mut sublayer.op else {
            unreachable!()
        };
        let HistoryReads::Selected(Selection::Own(indexer)) = &mut attention.history[1].reads else {
            unreachable!()
        };
        indexer.keys = IndexerKeys::Shared { source, part: 1 };
        indexer.candidates = crate::Candidates::Restrict { source, part: 1 };
        attention.history[1].rows = HistoryRows::Shared { source, part: 1 };
        sublayer
    }

    fn indexer_of(model: &mut ModelDefinition, block: usize) -> &mut Indexer {
        match &mut attention_of(model, block).history[1].reads {
            HistoryReads::Selected(Selection::Own(indexer)) => indexer,
            _ => unreachable!(),
        }
    }

    #[test]
    fn identity_rows_reindex_layers_and_candidate_pools_validate() {
        let mut model = definition(vec![compressing("a"), reusing("b"), reindexing("c")]);
        // Ratio 1: the rows are the tokens' own.
        let HistoryRows::Own(RowProducer::Pooled(rows)) = &mut attention_of(&mut model, 0).history[1].rows else {
            unreachable!()
        };
        (rows.rate, rows.pool) = (1, Pool::Identity);
        // A pool of exactly the restricted top's 512 units.
        indexer_of(&mut model, 0).candidates = crate::Candidates::Produce { blocks: 8, block_units: 64 };
        model.validate().unwrap();
        let invalid = |change: &dyn Fn(&mut ModelDefinition), error: &str| {
            let mut model = model.clone();
            change(&mut model);
            assert_eq!(model.validate().unwrap_err().to_string(), error);
        };
        let earlier = SublayerIndex {
            block: 1,
            sublayer: 0,
        };
        invalid(
            &|model| indexer_of(model, 0).candidates = crate::Candidates::Produce { blocks: 4, block_units: 64 },
            "a candidate pool of 256 units is smaller than its restricted indexer's top 512",
        );
        invalid(
            &|model| indexer_of(model, 0).candidates = crate::Candidates::All,
            "a restricted indexer names no earlier candidate pool over the same rows",
        );
        invalid(
            &|model| indexer_of(model, 2).candidates = crate::Candidates::Restrict { source: earlier, part: 1 },
            "a restricted indexer names no earlier candidate pool over the same rows",
        );
        invalid(
            &|model| indexer_of(model, 2).keys = IndexerKeys::Shared { source: earlier, part: 1 },
            "shared indexer keys name no earlier indexer keeping its own over the same rows",
        );
        invalid(
            &|model| indexer_of(model, 2).candidates = crate::Candidates::Produce { blocks: 8, block_units: 64 },
            "a candidate pool is produced over its own pooled rows and keys, in blocks of units",
        );
    }

    /// A two-stream, rank-3 site over hidden 4.
    fn low_rank(name: &str, inject: bool) -> LowRankGatedSite {
        LowRankGatedSite {
            norm: rms(&format!("{name}.hc_norm"), 8),
            down: weight(&format!("{name}.hc_down"), &[3, 8]),
            up: weight(&format!("{name}.hc_up"), &[8, 3]),
            inject: inject.then(|| weight(&format!("{name}.hc_inject"), &[2, 8])),
        }
    }

    fn site(name: &str, inject: bool) -> HyperConnectionWeights {
        HyperConnectionWeights::LowRankGated(low_rank(name, inject))
    }

    fn exit(site: LowRankGatedSite) -> ExitNorm {
        ExitNorm::HyperConnectionHead(HyperConnectionExit {
            site: ExitSite::Own(site),
            norm: None,
        })
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
                    token_map: None,
                    fill: 7,
                    boundary: Some(7),
                    media: NGramMedia::Hashed,
                },
                width: 2,
                table: weight("ple.table", &[12, 2]),
                key: weight("ple.key", &[8, 4]),
                value: weight("ple.value", &[4, 4]),
                gate: NGramGate::Norms {
                    key: weight("ple.norm_key", &[8]),
                    query: weight("ple.norm_query", &[8]),
                },
                epsilon: 1e-6,
                mixing: Some(NGramConvolution {
                    norm: weight("ple.norm_conv", &[8]),
                    taps: weight("ple.conv", &[8, 4]),
                    kernel: 4,
                    dilation: 3,
                }),
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
        model.decoder.exit.norm = exit(low_rank("output", false));
        model
    }

    fn mixed(mut sublayer: Sublayer, name: &str) -> Sublayer {
        sublayer.input = InputNorm::HyperConnectionMix(HyperConnectionInput {
            site: site(name, true),
            norm: None,
        });
        sublayer.output = OutputForm::Scatter;
        sublayer
    }

    #[test]
    fn hyper_connections_with_selected_history_and_ngram_input_validate() {
        let mut selected = attention("a", owned("a", TokenDomain::Token));
        let Operator::Attention(operator) = &mut selected.op else {
            unreachable!()
        };
        // Qwen4exp's form: mean-pooled keys over blocks of 4 tokens, the two
        // best blocks.
        operator.history[0].reads = HistoryReads::Selected(Selection::Own(Box::new(Indexer {
            heads: 2,
            width: 2,
            query: IndexerQuery {
                input: IndexerInput::Hidden,
                projection: weight("a.indexer.q", &[4, 4]),
                norm: HeadNorm::Rms(rms("a.indexer.q_norm", 2)),
                rounding: RowRounding::None,
            },
            keys: IndexerKeys::Pooled(PooledRows {
                rate: 4,
                projection: weight("a.indexer.k", &[2, 4]),
                pool: Pool::Mean,
                norm: rms("a.indexer.k_norm", 2),
                rounding: RowRounding::None,
            }),
            head_weights: HeadWeights::Uniform,
            scale: 2f64.powf(-0.5),
            top: 2,
            candidates: crate::Candidates::All,
        })));
        let model = hyper_connected(vec![ngram(), mixed(selected, "a")]);
        model.validate().unwrap();
        assert_eq!(model.deferred_forms(), []);
    }

    #[test]
    fn sublayers_must_match_the_residual_form() {
        let plain = attention("a", owned("a", TokenDomain::Token));
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
        exit_injects.decoder.exit.norm = exit(low_rank("output", true));
        assert_eq!(rejects(exit_injects), "decoder output norm injects into no stream");
    }

    /// Two streams over hidden 4 under single-pass Sinkhorn mixing: each
    /// sublayer's site forms (2 + 2)·2 = 8 mixes of the 8 stream values.
    fn sinkhorn(sublayers: Vec<Sublayer>) -> ModelDefinition {
        let mut model = definition(sublayers);
        model.decoder.residual = ResidualForm::HyperConnections(HyperConnections {
            streams: 2,
            mixing: HyperConnectionMixing::Sinkhorn {
                iterations: 20,
                epsilon: 1e-6,
                norm_epsilon: 1e-20,
            },
        });
        model.decoder.exit.norm = ExitNorm::HyperConnectionHead(HyperConnectionExit {
            site: ExitSite::PreviousSite,
            norm: Some(rms("output_norm", 4)),
        });
        model
    }

    fn sinkhorn_site(name: &str, mixes: u64, scales: u64) -> HyperConnectionWeights {
        HyperConnectionWeights::Sinkhorn(SinkhornSite {
            function: weight(&format!("{name}.hc_fn"), &[mixes, 8]),
            base: weight(&format!("{name}.hc_base"), &[mixes]),
            scale: weight(&format!("{name}.hc_scale"), &[scales]),
        })
    }

    fn sinkhorn_mixed(mut sublayer: Sublayer, name: &str) -> Sublayer {
        sublayer.input = InputNorm::HyperConnectionMix(HyperConnectionInput {
            site: sinkhorn_site(name, 8, 3),
            norm: Some(rms(&format!("{name}.norm"), 4)),
        });
        sublayer.output = OutputForm::Scatter;
        sublayer
    }

    #[test]
    fn single_pass_sinkhorn_sites_validate_against_their_mixing() {
        let plain = attention("a", owned("a", TokenDomain::Token));
        let rejects = |model: ModelDefinition| model.validate().unwrap_err().to_string();
        let model = sinkhorn(vec![sinkhorn_mixed(plain.clone(), "a")]);
        model.validate().unwrap();
        assert_eq!(model.deferred_forms(), []);
        // Only Sinkhorn mixing has a previous site's pre at the exit.
        let mut previous = hyper_connected(vec![mixed(plain.clone(), "a")]);
        previous.decoder.exit.norm = model.decoder.exit.norm.clone();
        assert_eq!(
            rejects(previous),
            "decoder output norm reads the previous site's coefficients without Sinkhorn mixing"
        );
        // A low-rank site under Sinkhorn mixing, and the reverse.
        assert_eq!(
            rejects(sinkhorn(vec![mixed(plain.clone(), "a")])),
            "sublayer input norm's site disagrees with the decoder's hyper-connection mixing"
        );
        assert_eq!(
            rejects(hyper_connected(vec![sinkhorn_mixed(plain.clone(), "a")])),
            "sublayer input norm's site disagrees with the decoder's hyper-connection mixing"
        );
        // A sublayer's site forms (2 + S)·S mixes.
        let mut short = sinkhorn_mixed(plain, "a");
        let InputNorm::HyperConnectionMix(input) = &mut short.input else {
            unreachable!()
        };
        input.site = sinkhorn_site("a", 2, 1);
        assert!(sinkhorn(vec![short]).validate().is_err());
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
        let previous = vec![hash.fill; hash.context as usize];
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
                .rposition(|token| Some(*token) == hash.boundary)
                .map_or(-1, |index| index as i64);
            let in_segment = at as i64 - (previous + 1);
            let shifted = (0..=context)
                .map(|shift| match in_segment >= shift as i64 {
                    true => history[at - shift],
                    false => hash.fill,
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
            token_map: None,
            fill: 7,
            boundary: Some(7),
            media: NGramMedia::Hashed,
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
            token_map: None,
            fill: 2,
            boundary: Some(2),
            media: NGramMedia::Hashed,
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

    /// DeepSeek V4.1's rule, over compressed token ids: every token hashes as its mapped id, a
    /// predecessor before the sequence reads the fill, and no token cuts.
    #[test]
    fn mapped_rows_read_the_fill_before_the_sequence_and_never_cut() {
        let map = (0..40u32).map(|token| (token * 7) % 13).collect::<Vec<_>>();
        let hash = NGramHash {
            context: 3,
            heads_per_order: 2,
            multipliers: vec![1_000_003, 999_983, 1_000_033, 1_000_037],
            sizes: (0..6).map(|head| 101 + 2 * head).collect(),
            offsets: (0..6).map(|head| head * 120).collect(),
            token_map: Some(map.clone()),
            fill: 2,
            boundary: None,
            media: NGramMedia::Hashed,
        };
        let mut state = 5u64;
        let tokens = (0..40)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) % 40
            })
            .collect::<Vec<_>>();
        let compressed = tokens.iter().map(|token| u64::from(map[*token as usize]));
        let shifted = std::iter::repeat_n(hash.fill, 3).chain(compressed).collect::<Vec<_>>();
        let mut expected = Vec::new();
        for at in 3..shifted.len() {
            for order in 2..=4 {
                let mixed = (0..order).fold(0, |mixed, distance| {
                    mixed ^ shifted[at - distance] * hash.multipliers[distance]
                });
                for head in (order - 2) * 2..(order - 1) * 2 {
                    expected.push(hash.offsets[head] + mixed % hash.sizes[head]);
                }
            }
        }
        let whole = hash.rows(&[], &tokens);
        assert_eq!(whole, expected);
        for split in [1, 2, 3, 4, 17] {
            let mut chunked = hash.rows(&[], &tokens[..split]);
            chunked.extend(hash.rows(&tokens[..split], &tokens[split..]));
            assert_eq!(chunked, whole, "split at {split}");
        }
    }

    /// A map covers the decoder's vocabulary, and the fill and every
    /// product are of its hashed ids.
    #[test]
    fn a_token_map_bounds_the_hashed_ids() {
        let model = |edit: &dyn Fn(&mut NGramInput)| {
            let mut sublayer = ngram();
            let Operator::NGramInput(input) = &mut sublayer.op else {
                unreachable!()
            };
            edit(input);
            hyper_connected(vec![sublayer]).validate()
        };
        let vocabulary = definition(vec![]).decoder.vocabulary as usize;
        let mapped = |input: &mut NGramInput| {
            input.hash.token_map = Some(vec![3; vocabulary]);
            input.hash.boundary = None;
            input.hash.fill = 3;
        };
        model(&mapped).unwrap();
        for (label, edit) in [
            ("short map", &(|input: &mut NGramInput| {
                mapped(input);
                input.hash.token_map = Some(vec![3; vocabulary - 1]);
            }) as &dyn Fn(&mut NGramInput)),
            ("fill past the hashed ids", &|input| {
                mapped(input);
                input.hash.fill = 4;
            }),
            ("inexact products", &|input| {
                mapped(input);
                input.hash.multipliers[1] = u64::MAX / 3;
            }),
        ] {
            assert!(model(edit).is_err(), "{label}");
        }
        // Without a convolution, under one product weight.
        model(&|input| {
            input.mixing = None;
            input.gate = NGramGate::Product(weight("ple.gate", &[8]));
        })
        .unwrap();
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

    /// A grid gives a tile only when it divides the matrix into power-of-two
    /// tiles: a partial last tile (300 rows in tiles of 32, 10 grid rows)
    /// or a non-power-of-two inferred tile (192 rows in tiles of 128, 2 grid
    /// rows, inferred 96) is refused rather than misread.
    #[test]
    fn a_scale_grid_tiles_only_by_exact_powers_of_two() {
        assert_eq!(ImportTransform::grid_tile(&[64, 256], &[2, 8]), Some([32, 32]));
        assert_eq!(ImportTransform::grid_tile(&[64, 256], &[64, 8]), Some([1, 32]));
        assert_eq!(ImportTransform::grid_tile(&[64, 256], &[1, 2]), Some([64, 128]));
        for (shape, grid) in [
            ([300, 4096], [10, 128]),
            ([192, 256], [2, 8]),
            ([8, 64], [3, 2]),
            ([64, 256], [0, 8]),
        ] {
            assert_eq!(ImportTransform::grid_tile(&shape, &grid), None, "{shape:?} {grid:?}");
        }
        assert_eq!(ImportTransform::grid_tile(&[64, 256, 1], &[2, 8]), None);
        // A stack's grid tiles every matrix alike, over the same matrices.
        assert_eq!(ImportTransform::grid_tile(&[3, 64, 256], &[3, 2, 8]), Some([32, 32]));
        assert_eq!(ImportTransform::grid_tile(&[3, 64, 256], &[2, 2, 8]), None);
        assert_eq!(ImportTransform::grid_tile(&[3, 64, 256], &[2, 8]), None);
        assert_eq!(ImportTransform::grid_tile(&[2, 3, 64, 256], &[2, 3, 2, 8]), None);
        let tiled = |tile| WeightDescriptor {
            name: "w".into(),
            shape: vec![64, 256],
            transforms: vec![ImportTransform::ScaleBlocks {
                tensor: "w.scale".into(),
                tile,
                codes: ScaleCodes::Ue8m0,
            }],
        };
        assert_eq!(tiled([32, 32]).transformed_shape(&[64, 256]).unwrap(), [64, 256]);
        assert!(tiled([16, 48]).transformed_shape(&[64, 256]).is_err());
        assert!(tiled([0, 32]).transformed_shape(&[64, 256]).is_err());
        assert_eq!(tiled([32, 32]).transformed_shape(&[3, 64, 256]).unwrap(), [3, 64, 256]);
        assert!(tiled([32, 32]).transformed_shape(&[2, 3, 64, 256]).is_err());
    }

    /// An I8 grid holds UE8M0 bytes and an F32 grid F32 scales; a grid of
    /// any other type holds no codes.
    #[test]
    fn scale_codes_are_named_by_their_stored_type() {
        use magnitude_artifacts::gguf::Encoding;
        for encoding in Encoding::ALL {
            let codes = match encoding {
                Encoding::I8 => Some(ScaleCodes::Ue8m0),
                Encoding::F32 => Some(ScaleCodes::F32),
                _ => None,
            };
            assert_eq!(ScaleCodes::of_stored(encoding), codes, "{encoding:?}");
        }
    }
}
