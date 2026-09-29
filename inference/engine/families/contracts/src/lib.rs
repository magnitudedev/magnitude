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
    InputNorm, KeyValue, LatentAttention, LatentExperts, LayerNorm, MediaRowAttention, Operator,
    OutputForm, PerLayerEntry, PerLayerInput, RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary,
    RotaryDivisors, RotaryPair, RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction,
    SharedExpert, SharedExpertGate,
    ShortConv, StateSpace, Sublayer, SublayerIndex, UnweightedRms, ValueNorm, ValueSource,
};
pub use draft::{
    BlockLayout, CandidateSelector, ConfidenceHead, DraftDefinition, DraftEmbedding, DraftMethod,
    DraftVariant, DynamicConvolution, LayerConvolutions, MarkovHead, TapPoint,
};
pub use family::{FamilyError, FamilyInputAdapter, MarkerTokens, ModelFamily};
pub use images::{ImageMarkers, SequentialImageInput, SpatialControls};
pub use inputs::{
    InputPreparationError, ModelInputAdapter, PreparedModelInput, PreparedVisionInput, TokenPlan,
    VisionSpatialControls,
};
pub use vision::{
    CellReduction, LinearClamp, LinearPart, MergerStage, NormPart, PositionSampling, Standardize,
    VisionActivation, VisionAttention, VisionAttentionScale, VisionAttentionSpan, VisionBlock,
    VisionDescription, VisionFeedForward, VisionLinear, VisionLinearSite, VisionMerger,
    VisionNorm, VisionNormSite, VisionPositions, VisionPreprocessing, VisionResampling,
    VisionResize, VisionStem, VisionUp, VisionWeight,
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
                ImportTransform::ScaleByTensor { tensor } => {
                    if tensor.is_empty() || *tensor == self.name {
                        return Err(DefinitionError::new(format!(
                            "scale tensor of {:?} must name another stored tensor",
                            self.name
                        )));
                    }
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
    /// Multiply every matrix of a stack (the whole tensor when it is not a
    /// stack) by the matching element of the stored F32 tensor `tensor`,
    /// shaped `[1]` or `[stack]`. A container format's second-level scale
    /// (NVFP4) stays a separate factor of the resident weight, so it is exact
    /// for block encodings too.
    ScaleByTensor { tensor: String },
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
            Self::Flatten => {}
            Self::ScaleByTensor { tensor } => tensor.hash(state),
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
    TargetBranch { sublayer: SublayerIndex, branch: u32 },
    /// Weights of a draft head block outside its sublayers.
    HeadBlock(u32),
    /// A sublayer of a draft head block (`block` is the head block).
    HeadSublayer(SublayerIndex),
    HeadBranch { sublayer: SublayerIndex, branch: u32 },
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
    HyperConnectionMix,
    Query,
    QueryGate,
    AttentionGate,
    Key,
    Value,
    QueryNorm,
    KeyNorm,
    AttentionOutput,
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
    RotaryDivisors,
    OutputNorm,
    Output,
    HeadEmbeddingNorm,
    HeadHiddenNorm,
    HeadCombine,
    /// A draft's projection of the concatenated target taps.
    DraftFusion,
    DraftFusionNorm,
    MarkovEmbedding,
    MarkovProjection,
    ConfidenceWeight,
    ConfidenceBias,
    /// DFlash2: a draft sublayer's convolution base `[2, kernel, hidden]`
    /// and coefficient projection (scope `DraftSublayer`).
    ConvolutionBase,
    ConvolutionProjection,
    /// DFlash2's candidate selector: hidden projection and codebooks.
    SelectorHidden,
    SelectorPredecessor,
    SelectorSuccessor,
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

/// One draft block following the target decoder: it combines the embedding
/// of the next token with the target's hidden row, runs one decoder block and
/// normalizes its output for the target's vocabulary projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadBlock {
    pub embedding_norm: RmsNorm,
    pub hidden_norm: RmsNorm,
    /// `[hidden, 2·hidden]`.
    pub combine: WeightDescriptor,
    pub block: Block,
    pub output_norm: ExitNorm,
}

/// An embedded draft head (MTP): blocks that read the target's embedding and
/// vocabulary projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Head {
    pub blocks: Vec<HeadBlock>,
}

impl Head {
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    pub fn validate(&self, decoder: &Decoder, coordinate_axes: u8) -> Result<(), DefinitionError> {
        if self.blocks.is_empty() {
            return Err(DefinitionError::new("draft head has no blocks"));
        }
        let combined_hidden = checked_product(&[2, decoder.hidden])?;
        let context = decoder::Context {
            hidden: decoder.hidden,
            coordinate_axes,
            per_layer: None,
        };
        for block in &self.blocks {
            decoder::validate_rms(&block.embedding_norm, decoder.hidden, "draft head norm")?;
            decoder::validate_rms(&block.hidden_norm, decoder.hidden, "draft head norm")?;
            expect_shape(
                &block.combine,
                &[decoder.hidden, combined_hidden],
                "draft head input projection",
            )?;
            decoder::validate_blocks(std::slice::from_ref(&block.block), &context)?;
            decoder::validate_exit_norm(&block.output_norm, decoder.hidden, "draft head norm")?;
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
                decoder::blocks_deferred_forms(std::slice::from_ref(&block.block), &mut forms);
                if matches!(block.output_norm, ExitNorm::HyperConnectionHead(_)) {
                    forms.push(DeferredForm::HyperConnectionHead);
                }
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
            inputs: InputSemantics {
                coordinate_axes: 1,
            },
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
    fn deferred_forms_are_named_not_rejected_by_the_contract() {
        let mut model = definition(vec![attention(
            "a",
            owned("a", HistoryDomain::Block { rate: 4 }),
        )]);
        model.decoder.residual = ResidualForm::HyperConnections(HyperConnections {
            streams: 4,
            mixing: HyperConnectionMixing::Sinkhorn { iterations: 20 },
        });
        assert!(model.validate().is_ok());
        assert_eq!(
            model.deferred_forms(),
            [DeferredForm::HyperConnections, DeferredForm::BlockHistory]
        );
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
            }],
        };
        assert_eq!(scaled.transformed_shape(&[2, 4, 8]).unwrap(), [2, 4, 8]);
    }
}
