#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityPlan {
    pub(super) max_draft_proposals: Option<u8>,
    pub(super) vision: bool,
}

/// The most proposals one draft-head transaction chains. Every width up to
/// the planned maximum has sealed head graphs, so this bounds load work.
pub const MAX_DRAFT_PROPOSALS: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedMethod {
    Plain,
    Mtp {
        greedy_proposals: u8,
        sampled_proposals: u8,
    },
    /// A separate DFlash/DSpark draft, greedy and sampled alike: a round
    /// carries at most `proposals` draft rows, and each block pass drafts
    /// exactly `generated` of them (none when 0). Both are fixed per load.
    DFlash { proposals: u8, generated: u8 },
}

impl PlannedMethod {
    /// The most draft rows a verify slot carries after its anchor: the rows a
    /// recurrent bank records past its published state.
    pub fn draft_rows(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => usize::from(greedy_proposals.max(sampled_proposals)),
            Self::DFlash { proposals, .. } => usize::from(proposals),
        }
    }

    /// The proposals each block pass of a separate draft generates: 0 when
    /// the method drafts no block.
    pub fn block_proposals(self) -> u8 {
        match self {
            Self::DFlash { generated, .. } => generated,
            Self::Plain | Self::Mtp { .. } => 0,
        }
    }

    /// How the method's drafter drafts; `None` for plain generation.
    pub fn draft_form(self) -> Option<crate::DraftForm> {
        match self {
            Self::Plain => None,
            Self::Mtp { .. } => Some(crate::DraftForm::Chained),
            Self::DFlash { .. } => Some(crate::DraftForm::Block),
        }
    }

    /// Whether the method drafts with the definition's drafter.
    pub fn drafts(self) -> bool {
        !matches!(self, Self::Plain)
    }
}

impl CapabilityPlan {
    pub fn max_draft_proposals(&self) -> Option<u8> {
        self.max_draft_proposals
    }

    pub fn supports_vision(&self) -> bool {
        self.vision
    }
}
