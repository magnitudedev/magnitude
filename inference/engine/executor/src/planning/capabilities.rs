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
    /// carries at most `proposals` draft rows. Each block pass drafts exactly
    /// `generated` of them (none when 0), or with `ladder` any width of
    /// [`draft_ladder`] up to `generated`. All are fixed per load.
    DFlash {
        proposals: u8,
        generated: u8,
        ladder: bool,
    },
}

/// The draft widths worth a round of their own, up to `widest`: those whose
/// verification (the drafts and their anchor) fills a launch row class, and
/// `widest` itself. Any other width pays for the next class's rows and
/// commits fewer tokens.
pub fn draft_ladder(widest: usize) -> Vec<usize> {
    let mut ladder = (1..=widest)
        .filter(|width| (width + 1).is_power_of_two())
        .collect::<Vec<_>>();
    if widest > 0 && ladder.last() != Some(&widest) {
        ladder.push(widest);
    }
    ladder
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

    /// The draft widths a load of this method decodes at, any of which its
    /// adaptive controller may run a round at (`draft_control`): none, a
    /// plain step, and each width of the method's ladder; nothing beside
    /// the plain step without drafting.
    pub fn draft_widths(self) -> Vec<u64> {
        let ladder = match self {
            Self::Plain => return Vec::new(),
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
            } => draft_ladder(usize::from(greedy_proposals.max(sampled_proposals)))
                .into_iter()
                .map(|width| width as u64)
                .collect::<Vec<_>>(),
            Self::DFlash { .. } => self.block_widths().into_iter().map(u64::from).collect(),
        };
        std::iter::once(0).chain(ladder).collect()
    }

    /// The proposal counts a block pass of a separate draft may generate, in
    /// increasing order: none when the method drafts no block.
    pub fn block_widths(self) -> Vec<u8> {
        match self {
            Self::DFlash {
                generated,
                ladder: true,
                ..
            } => draft_ladder(usize::from(generated))
                .into_iter()
                .map(|width| width as u8)
                .collect(),
            Self::DFlash { generated, .. } => {
                (generated > 0).then_some(generated).into_iter().collect()
            }
            Self::Plain | Self::Mtp { .. } => Vec::new(),
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
