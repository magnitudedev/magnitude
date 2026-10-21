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
        widths: DraftWidths,
    },
    /// A separate DFlash/DSpark draft, greedy and sampled alike: a round
    /// carries at most `proposals` draft rows. Each block pass drafts a width
    /// of [`draft_ladder`] up to the policy's widest ([`Self::widest`]). All
    /// are fixed per load.
    DFlash { proposals: u8, widths: DraftWidths },
}

/// How a load's rounds choose their draft width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftWidths {
    /// The adaptive controller runs a round at any width of the method's
    /// ladder up to its proposals (`draft_control`).
    Adaptive,
    /// A fixed budget's generated width (none when 0) whenever the request
    /// can take it, and a narrower width of its ladder only in a request's
    /// last rounds.
    Fixed(u8),
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
                ..
            } => usize::from(greedy_proposals.max(sampled_proposals)),
            Self::DFlash { proposals, .. } => usize::from(proposals),
        }
    }

    /// The widest draft a round of this method generates: the adaptive
    /// controller's widest, its proposals, or a fixed budget's width; 0
    /// without drafting.
    pub fn widest(self) -> u8 {
        match self {
            Self::Plain => 0,
            Self::Mtp {
                widths: DraftWidths::Fixed(generated),
                ..
            }
            | Self::DFlash {
                widths: DraftWidths::Fixed(generated),
                ..
            } => generated,
            Self::Mtp {
                greedy_proposals,
                sampled_proposals,
                widths: DraftWidths::Adaptive,
            } => greedy_proposals.max(sampled_proposals),
            Self::DFlash {
                proposals,
                widths: DraftWidths::Adaptive,
            } => proposals,
        }
    }

    /// The draft widths whose decode rounds the load's tuning workload
    /// weighs, alike: a plain step, then each width of the ladder an
    /// adaptive controller may run a round at, or a fixed budget's width
    /// alone (its narrower widths serve only a request's last rounds: a
    /// block drafter prepares them, at their defaults); nothing beside the
    /// plain step without drafting.
    pub fn draft_widths(self) -> Vec<u64> {
        let widths = match self {
            Self::Plain => return Vec::new(),
            Self::Mtp {
                widths: DraftWidths::Adaptive,
                ..
            }
            | Self::DFlash {
                widths: DraftWidths::Adaptive,
                ..
            } => draft_ladder(usize::from(self.widest())),
            Self::Mtp {
                widths: DraftWidths::Fixed(generated),
                ..
            }
            | Self::DFlash {
                widths: DraftWidths::Fixed(generated),
                ..
            } => (generated > 0).then_some(usize::from(generated)).into_iter().collect(),
        };
        std::iter::once(0)
            .chain(widths.into_iter().map(|width| width as u64))
            .collect()
    }

    /// The proposal counts a block pass of a separate draft may generate, in
    /// increasing order: every width of the ladder up to the widest, so a
    /// round near a request's end drafts what fits; none when the method
    /// drafts no block.
    pub fn block_widths(self) -> Vec<u8> {
        match self {
            Self::DFlash { .. } => draft_ladder(usize::from(self.widest()))
                .into_iter()
                .map(|width| width as u8)
                .collect(),
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

#[cfg(test)]
mod tests {
    use super::{DraftWidths, PlannedMethod};

    /// A block pass drafts any width of the ladder up to its widest, so a
    /// round near a request's end still drafts what fits; the workload
    /// weighs a fixed budget's own width alone.
    #[test]
    fn a_separate_draft_prepares_the_ladder_up_to_its_widest_width() {
        let fixed = |generated| PlannedMethod::DFlash {
            proposals: 7,
            widths: DraftWidths::Fixed(generated),
        };
        let adaptive = PlannedMethod::DFlash {
            proposals: 7,
            widths: DraftWidths::Adaptive,
        };
        assert_eq!(adaptive.block_widths(), [1, 3, 7]);
        assert_eq!(fixed(7).block_widths(), [1, 3, 7]);
        assert_eq!(fixed(5).block_widths(), [1, 3, 5]);
        assert_eq!(fixed(1).block_widths(), [1]);
        assert!(fixed(0).block_widths().is_empty());
        assert_eq!(adaptive.draft_widths(), [0, 1, 3, 7]);
        assert_eq!(fixed(5).draft_widths(), [0, 5]);
        assert_eq!(fixed(0).draft_widths(), [0]);
        let mtp = |widths| PlannedMethod::Mtp {
            greedy_proposals: 3,
            sampled_proposals: 1,
            widths,
        };
        assert_eq!(mtp(DraftWidths::Adaptive).draft_widths(), [0, 1, 3]);
        assert_eq!(mtp(DraftWidths::Fixed(2)).draft_widths(), [0, 2]);
    }
}
