use crate::lowered_ir::DecisionKind;

/// Closed migration inventory. Adding a legacy decision kind requires an
/// explicit entry here; coverage cannot be hidden behind a wildcard baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DecisionClass {
    Construct, Stream, OutputGroup, OutputRemainders, GroupEpilogue,
    MatrixPanel, Representation, PacketDecode, ReductionSegments,
    Intermediate, ReductionBranch, Reduction, ParallelFusion, StreamFusion,
    FoldOperand, FoldState, FoldTraversal, FoldPreparation, FoldCoefficients,
    FoldWords, PacketDecoder, ReductionInput, ReductionFusion, RangeFusion,
    Producer, Load,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Migration { Retained, CoverageRequired(&'static str) }

impl DecisionClass {
    pub fn of(kind: &DecisionKind) -> Self {
        match kind {
            DecisionKind::Construct { .. } => Self::Construct,
            DecisionKind::Stream { .. } => Self::Stream,
            DecisionKind::OutputGroup { .. } => Self::OutputGroup,
            DecisionKind::OutputRemainders { .. } => Self::OutputRemainders,
            DecisionKind::GroupEpilogue { .. } => Self::GroupEpilogue,
            DecisionKind::MatrixPanel { .. } => Self::MatrixPanel,
            DecisionKind::Representation { .. } => Self::Representation,
            DecisionKind::PacketDecode { .. } => Self::PacketDecode,
            DecisionKind::ReductionSegments { .. } => Self::ReductionSegments,
            DecisionKind::Intermediate { .. } => Self::Intermediate,
            DecisionKind::ReductionBranch { .. } | DecisionKind::ReductionFrontier { .. } => Self::ReductionBranch,
            DecisionKind::Reduction { .. } => Self::Reduction,
            DecisionKind::ParallelFusion { .. } => Self::ParallelFusion,
            DecisionKind::StreamFusion { .. } => Self::StreamFusion,
            DecisionKind::FoldOperand { .. } => Self::FoldOperand,
            DecisionKind::FoldState { .. } => Self::FoldState,
            DecisionKind::FoldTraversal { .. } => Self::FoldTraversal,
            DecisionKind::FoldPreparation { .. } => Self::FoldPreparation,
            DecisionKind::FoldCoefficients { .. } => Self::FoldCoefficients,
            DecisionKind::FoldWords { .. } => Self::FoldWords,
            DecisionKind::PacketDecoder { .. } => Self::PacketDecoder,
            DecisionKind::ReductionInput { .. } => Self::ReductionInput,
            DecisionKind::ReductionFusion { .. } => Self::ReductionFusion,
            DecisionKind::RangeFusion { .. } => Self::RangeFusion,
            DecisionKind::Producer { .. } => Self::Producer,
        }
    }
    pub fn migration(self) -> Migration {
        match self {
            Self::Construct | Self::Stream | Self::Reduction | Self::ReductionSegments
                | Self::FoldOperand | Self::FoldState | Self::FoldTraversal | Self::FoldPreparation
                | Self::FoldCoefficients | Self::FoldWords | Self::ReductionInput
                | Self::MatrixPanel => Migration::Retained,
            Self::OutputGroup | Self::OutputRemainders | Self::GroupEpilogue =>
                Migration::CoverageRequired("retain output grouping and epilogue quotient/remainder topology"),
            Self::Representation | Self::PacketDecode | Self::PacketDecoder =>
                Migration::CoverageRequired("retain encoded/decoded packet producer and decoder alternatives"),
            Self::ReductionBranch => Migration::Retained,
            Self::Intermediate | Self::Producer =>
                Migration::CoverageRequired("retain projected/recomputed producers and publication lifetimes"),
            Self::ParallelFusion | Self::StreamFusion | Self::ReductionFusion | Self::RangeFusion =>
                Migration::CoverageRequired("retain fusion compatibility, effects and conditional publications"),
            Self::Load => Migration::CoverageRequired("retain normalized load ownership with complete guarded lifetimes"),
        }
    }
}
